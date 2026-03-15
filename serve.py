#!/usr/bin/env python3
"""Serve the PPM Compliance Checker locally. Run: python3 serve.py

DEPENDENCIES
  PDF support:  pip install pdfplumber   (recommended — handles tables, columns)
                pip install pypdf        (fallback — simpler, lighter)
  DOCX support: built-in stdlib (no install needed)
  TXT support:  built-in stdlib (no install needed)

If neither PDF library is installed, PDF uploads show an install prompt.
All other functionality works without them.
"""
# WARNING: do not expose on public networks — no auth, no TLS.
import http.server, webbrowser, os, io, json, urllib.parse, zipfile, re, math, sys, socket
import xml.etree.ElementTree as ET

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB guard

_citation_re     = re.compile(r'"([^"]{50,})"')
_citation_noise  = re.compile(r'^[\s:]*\d+\.\s*(?:STATUS|COVERED|PARTIAL|MISSING):\s*', re.IGNORECASE)

# ── Frozen (PyInstaller) vs development paths ─────────────────────────────────
_FROZEN = getattr(sys, 'frozen', False)

# Read-only assets (bundled HTML, JSON) — inside the .app bundle when frozen
if _FROZEN:
    _BUNDLE_DIR = sys._MEIPASS
else:
    _BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Writable user-data directory (embeddings cache, models, etc.)
def _user_data_dir():
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    elif sys.platform == 'win32':
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
    else:
        base = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
    d = os.path.join(base, 'PPMChecker')
    os.makedirs(d, exist_ok=True)
    return d

_USER_DATA_DIR = _user_data_dir()

# ── Module-level embedding cache (in-memory + disk-backed) ────────────────────
_embed_mem_cache = {}   # {c_hash: {sentence: vector}}
_EMBED_CACHE_DIR = os.path.join(_USER_DATA_DIR, '.embeddings')

# ── llama.cpp model (loaded once at startup) ───────────────────────────────────
_llm        = None   # llama_cpp.Llama instance, or None
_llm_name   = None   # model filename shown in UI

def _find_model_file():
    """Search known locations for a .gguf model file."""
    search_dirs = [
        os.path.join(_USER_DATA_DIR, 'models'),  # ~/Library/…/PPMChecker/models/
    ]
    if _FROZEN:
        exe_dir = os.path.dirname(sys.executable)
        search_dirs += [
            os.path.join(exe_dir, '..', '..', '..', 'models'),  # macOS: sibling of .app
            os.path.join(exe_dir, '..', 'models'),               # Windows: sibling of PPMChecker/
            os.path.join(exe_dir, 'models'),                     # Windows onefile layout
        ]
    else:
        search_dirs += [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        ]
    # Prefer larger/more capable models first
    _PREFERRED = ['llama', 'mistral', 'gemma', 'phi', 'tinyllama']
    for d in dict.fromkeys(search_dirs):   # deduplicate preserving order
        if not os.path.isdir(d):
            continue
        gguf_files = [f for f in os.listdir(d) if f.lower().endswith('.gguf')]
        gguf_files.sort(key=lambda f: next(
            (i for i, p in enumerate(_PREFERRED) if p in f.lower()), 99))
        if gguf_files:
            return os.path.join(d, gguf_files[0])
    return None

def _init_llm():
    global _llm, _llm_name
    try:
        from llama_cpp import Llama
        model_path = _find_model_file()
        if not model_path:
            return
        _llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 2) - 1),
            verbose=False,
        )
        _llm_name = os.path.basename(model_path)
        print(f'[llm] Loaded {_llm_name}')
    except:
        pass  # llama_cpp unavailable or model missing — Ollama fallback

_init_llm()

def _load_embed_cache(c_hash):
    if c_hash in _embed_mem_cache:
        return _embed_mem_cache[c_hash]
    path = os.path.join(_EMBED_CACHE_DIR, f'{c_hash}.json')
    if os.path.exists(path):
        with open(path) as f:
            _embed_mem_cache[c_hash] = json.load(f)
    else:
        _embed_mem_cache[c_hash] = {}
    return _embed_mem_cache[c_hash]

def _save_embed_cache(c_hash):
    os.makedirs(_EMBED_CACHE_DIR, exist_ok=True)
    path = os.path.join(_EMBED_CACHE_DIR, f'{c_hash}.json')
    with open(path, 'w') as f:
        json.dump(_embed_mem_cache[c_hash], f)
# ──────────────────────────────────────────────────────────────────────────────

class PPMHandler(http.server.SimpleHTTPRequestHandler):

    _WORD_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/llm_status':
            self._llm_status()
            return
        if path in ('/', ''):
            path = '/PPM_Compliance_Checker.html'
        # When frozen, serve static files directly from bundle dir
        if _FROZEN:
            filename = path.lstrip('/')
            filepath = os.path.join(_BUNDLE_DIR, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    data = f.read()
                ext = filename.rsplit('.', 1)[-1].lower()
                ct = {'html': 'text/html', 'js': 'application/javascript',
                      'css': 'text/css', 'json': 'application/json'
                      }.get(ext, 'application/octet-stream')
                self.send_response(200)
                self.send_header('Content-Type', ct)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split('?')[0]

        if path == '/extract':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._json_error(400, 'No data received')
                return
            if length > MAX_UPLOAD_BYTES:
                self._json_error(413, 'File too large (max 20 MB)')
                return

            data = self.rfile.read(length)

            qs   = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = qs.get('name', [''])[0].lower()
            ext  = name.rsplit('.', 1)[-1] if '.' in name else ''

            try:
                if ext == 'pdf':
                    text = self._extract_pdf(data)
                elif ext == 'docx':
                    text = self._extract_docx(data)
                else:
                    self._json_error(400, f'Unsupported file type: .{ext}')
                    return
            except RuntimeError as e:
                self._json_error(500, str(e))
                return
            except Exception as e:
                self._json_error(500, f'Extraction failed: {e}')
                return

            if not text.strip():
                self._json_error(422, 'No text could be extracted — try copy-pasting instead')
                return

            self._json_ok({'text': text})

        elif path == '/llm_summary':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._json_error(400, 'No data received')
                return
            data = self.rfile.read(length)
            try:
                body = json.loads(data)
            except Exception:
                self._json_error(400, 'Invalid JSON')
                return
            self._llm_summary(body)

        elif path in ('/llm_check', '/llm_check_batch'):
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._json_error(400, 'No data received')
                return
            data = self.rfile.read(length)
            try:
                body = json.loads(data)
            except Exception:
                self._json_error(400, 'Invalid JSON')
                return
            if path == '/llm_check_batch':
                self._llm_check_batch(body)
            else:
                self._llm_check(body)

        else:
            self.send_error(404)

    def _extract_pdf(self, data):
        # Try pdfplumber first (better layout handling)
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        pages.append(text)
                result = '\n'.join(pages)
                if result.strip():
                    return result
        except ImportError:
            pass
        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            return '\n'.join(p.extract_text() or '' for p in reader.pages)
        except ImportError:
            raise RuntimeError(
                'No PDF library installed.\nRun: pip install pdfplumber\nThen restart serve.py.'
            )

    def _extract_docx(self, data):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise RuntimeError('File does not appear to be a valid .docx')
        xml  = zf.read('word/document.xml')
        root = ET.fromstring(xml)
        W    = self._WORD_NS
        lines = []
        for para in root.iter(f'{{{W}}}p'):
            text = ''.join(r.text for r in para.iter(f'{{{W}}}t') if r.text)
            if text.strip():
                lines.append(text)
        return '\n'.join(lines)

    def _llm_status(self):
        # 1. llama.cpp (bundled) takes priority
        if _llm is not None:
            self._json_ok({'available': True, 'model': _llm_name,
                           'embed_model': None, 'backend': 'llama_cpp'})
            return
        # 2. Fall back to Ollama if running
        try:
            import urllib.request
            resp = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)
            data = json.loads(resp.read())
            models = [m['name'] for m in data.get('models', [])]
            # Exclude embedding-only models from chat list
            embed_model = next((m for m in models if 'embed' in m or 'nomic' in m), None)
            chat_models = [m for m in models if 'embed' not in m and 'nomic' not in m]
            # Prefer larger/better models — qwen2.5 > llama3.1 > llama3.3 > mistral > others
            _model_prefs = ['qwen2.5', 'llama3.1', 'llama3.3', 'mistral', 'llama3.2', 'phi3', 'phi-3']
            chat_model = next((m for p in _model_prefs for m in chat_models if p in m), None)
            if not chat_model:
                chat_model = chat_models[0] if chat_models else None
            self._json_ok({'available': bool(chat_model), 'model': chat_model,
                           'models': chat_models, 'embed_model': embed_model, 'backend': 'ollama'})
        except Exception:
            self._json_ok({'available': False, 'model': None,
                           'embed_model': None, 'backend': None})

    def _llm_check(self, payload):
        task_id       = payload['task_id']
        task_title    = payload['task_title']
        task_notes    = payload.get('task_notes', '')
        task_action   = payload.get('task_action', '')
        required_freq = payload.get('required_freq', '')
        contract_text = payload['contract_text']
        chat_model    = payload.get('model', 'phi3:mini')

        # ── 1. Split contract — filter out title lines and T&C boilerplate ──────
        _TC_NOISE = re.compile(
            r'\b(buyer|seller|purchaser|vendor|invoice|payment|warranty|warrantee|'
            r'defect\w*|liabilit\w*|indemnif\w*|intellectual property|confidential|'
            r'force majeure|governing law|jurisdiction|terminat\w*|arbitration|'
            r'purchase order|acceptance of goods|'
            r'title (?:to|in) (?:the )?goods|risk (?:of|in) (?:the )?goods|'
            r'terms\s+(?:and|&)\s+conditions|conditions of sale|subject to these|'
            r'these conditions|these terms|time of delivery|period of \d+|'
            r'material and workmanship|free from defects?|'
            r'breach of (?:contract|warranty)|waiver?s?|'
            r'acknowledges?\s+that|does not rely|reasonable access|'
            r'pro.?forma|credit check|account form|upfront|up.front|'
            r'\d+\s*days?\s+(?:credit|pending|net)|days? pending|'
            r'subsequent orders?|propert[yi])\b',
            re.IGNORECASE
        )
        # Fragment detector — sentences ending mid-clause on a dangling
        # article/preposition were cut from a larger sentence and carry no
        # standalone meaning (e.g. "…service and maintenance of the").
        _FRAGMENT_END = re.compile(
            r'\s+(?:the|of|for|to|a|an|and|or|but|pending|with|by|from|at|on)\s*$',
            re.IGNORECASE
        )
        _HAS_VERB = re.compile(
            r'\b(is|are|was|were|will|shall|should|must|may|can|could|would|'
            r'includ|ensur|provid|perform|carry|carri|test|inspect|check|service|'
            r'maintain|replac|clean|verif|report|visit|complet|cover|supply|install|'
            r'monitor|record|calibrat|operat|function|responsibl)\w*\b',
            re.IGNORECASE
        )
        all_sentences = [
            s.strip() for s in re.split(r'\n+|(?<=[.!?;])\s+(?=[A-Z])', contract_text)
            if len(s.strip().split()) >= 5 and len(s.strip()) < 600
            and not _TC_NOISE.search(s)
            and not _FRAGMENT_END.search(s.strip())
            and _HAS_VERB.search(s)
        ]
        if not all_sentences:
            self._json_error(422, 'No sentences found in contract')
            return

        # ── 2. Rank sentences by keyword overlap ──────────────────────────────
        task_tokens = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}',
                         task_title + ' ' + task_notes + ' ' + task_action[:300]))

        def keyword_score(sentence):
            words = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', sentence))
            return len(task_tokens & words)

        scored = sorted(enumerate(all_sentences), key=lambda x: keyword_score(x[1]), reverse=True)
        top_sentences = [s for _, s in scored[:3] if keyword_score(s) > 0]
        key_sentences = top_sentences

        # ── 3. Build LLM prompt with clear verdict definitions ────────────────
        clauses = '\n'.join(f'- {s}' for s in top_sentences) if top_sentences else '(none found)'
        action_hint = task_action[:300].strip()
        task_desc = task_title
        if action_hint:
            task_desc += f"\nTask requires: {action_hint}"
        if required_freq:
            task_desc += f"\nRequired frequency: {required_freq}"
        prompt = f"""You are a PPM compliance reviewer.
COVERED = contract explicitly describes this task. PARTIAL = contract references this type of work but doesn't clearly commit to this specific task. MISSING = no mention; use when in doubt.
Task: {task_desc}
Contract: {clauses}
Reply: one word (COVERED/PARTIAL/MISSING) then one sentence reason."""

        if _llm is not None:
            # ── llama.cpp path ────────────────────────────────────────────────
            output = _llm.create_chat_completion(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=60,
                temperature=0.1,
            )
            raw = output['choices'][0]['message']['content'].strip()
        else:
            # ── Ollama fallback (chat API) ────────────────────────────────────
            import urllib.request as _ur
            body = json.dumps({
                'model': chat_model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'options': {'temperature': 0.1, 'num_predict': 60, 'num_ctx': 2048}
            }).encode()
            req      = _ur.Request('http://localhost:11434/api/chat',
                                   data=body, headers={'Content-Type': 'application/json'})
            llm_resp = json.loads(_ur.urlopen(req, timeout=300).read())
            raw = llm_resp.get('message', {}).get('content', '').strip()

        # ── 8. Parse response + downgrade PARTIAL when reasoning shows no evidence
        _no_evidence = re.compile(
            r'does not (?:explicitly |clearly )?(?:mention|cover|include|specify|address)|'
            r'no mention|not mentioned|not specified|not covered|not explicitly|'
            r'cannot confirm|no evidence|unclear',
            re.IGNORECASE
        )
        lines     = [l.strip() for l in raw.split('\n') if l.strip()]
        status    = next((l for l in lines if l in ('COVERED', 'PARTIAL', 'MISSING')), None)
        reasoning = next((l for l in lines if l not in ('COVERED', 'PARTIAL', 'MISSING')), raw)

        if not status:
            self._json_error(422, f'LLM returned unparseable response: {raw[:200]}')
            return

        if status == 'PARTIAL' and _no_evidence.search(reasoning):
            status = 'MISSING'

        self._json_ok({
            'task_id':       task_id,
            'status':        status,
            'reasoning':     reasoning,
            'key_sentences': key_sentences,
        })

    def _llm_check_batch(self, payload):
        tasks         = payload['tasks']   # [{task_id, task_title, task_notes}, ...]
        contract_text = payload['contract_text']
        chat_model    = payload.get('model', 'phi3:mini')

        # ── 1. Split contract into sentences — exclude T&C boilerplate ──────
        _TC_NOISE = re.compile(
            r'\b(buyer|seller|purchaser|vendor|invoice|payment|warranty|warrantee|'
            r'defect\w*|liabilit\w*|indemnif\w*|intellectual property|confidential|'
            r'force majeure|governing law|jurisdiction|terminat\w*|arbitration|'
            r'purchase order|acceptance of goods|'
            r'title (?:to|in) (?:the )?goods|risk (?:of|in) (?:the )?goods|'
            r'terms\s+(?:and|&)\s+conditions|conditions of sale|subject to these|'
            r'these conditions|these terms|time of delivery|period of \d+|'
            r'material and workmanship|free from defects?|'
            r'breach of (?:contract|warranty)|waiver?s?|'
            r'acknowledges?\s+that|does not rely|reasonable access|'
            r'pro.?forma|credit check|account form|upfront|up.front|'
            r'\d+\s*days?\s+(?:credit|pending|net)|days? pending|'
            r'subsequent orders?|propert[yi])\b',
            re.IGNORECASE
        )
        # Fragment detector — sentences ending mid-clause on a dangling
        # article/preposition were cut from a larger sentence and carry no
        # standalone meaning (e.g. "…service and maintenance of the").
        _FRAGMENT_END = re.compile(
            r'\s+(?:the|of|for|to|a|an|and|or|but|pending|with|by|from|at|on)\s*$',
            re.IGNORECASE
        )
        # Heading/title detector — lines with no verb-like word are almost certainly
        # section headings or cost/reference lines, not scope-of-work sentences.
        _HAS_VERB = re.compile(
            r'\b(is|are|was|were|will|shall|should|must|may|can|could|would|'
            r'includ|ensur|provid|perform|carry|carri|test|inspect|check|service|'
            r'maintain|replac|clean|verif|report|visit|complet|cover|supply|install|'
            r'monitor|record|calibrat|operat|function|responsibl)\w*\b',
            re.IGNORECASE
        )
        all_sentences = [
            s.strip() for s in re.split(r'\n+|(?<=[.!?;])\s+(?=[A-Z])', contract_text)
            if len(s.strip()) > 20 and len(s.strip()) < 600
            and not _TC_NOISE.search(s)
            and not _FRAGMENT_END.search(s.strip())
            and _HAS_VERB.search(s)
        ]
        print(f"[llm_check_batch] {len(tasks)} tasks, {len(all_sentences)} sentences from contract")

        # ── 2. Per-task: pick top sentences by keyword overlap ────────────────
        def keyword_score(sentence, tokens):
            words = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', sentence))
            return len(tokens & words)

        # Position index for context-sentence lookup
        sent_pos = {s: i for i, s in enumerate(all_sentences)}

        task_top_sentences = {}
        llm_tasks   = []
        pre_missing = []

        for task in tasks:
            provided = [s for s in task.get('evidence_sentences', []) if s.strip()]
            if provided:
                # Use HTML's synonym-aware evidence; add one context sentence
                # (the sentence immediately before the best evidence hit, if available)
                top2 = list(provided[:3])
                best_pos = sent_pos.get(provided[0], -1)
                if best_pos > 0:
                    ctx = all_sentences[best_pos - 1]
                    if ctx not in top2:
                        top2 = [ctx] + top2
                top2 = top2[:3]  # keep max 3 to stay within token budget
            else:
                # Fallback: naive keyword re-filter
                tokens = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}',
                             task['task_title'] + ' '
                             + task.get('task_notes', '') + ' '
                             + task.get('task_action', '')[:300]))
                scored_sents = sorted(all_sentences,
                                      key=lambda s: keyword_score(s, tokens), reverse=True)
                top2 = [s for s in scored_sents[:3] if keyword_score(s, tokens) >= 1]
            task_top_sentences[task['task_id']] = top2
            if top2:
                llm_tasks.append(task)
            else:
                pre_missing.append(task)
        print(f"[llm_check_batch] {len(llm_tasks)} tasks have evidence, {len(pre_missing)} pre-MISSING")

        # ── 3. Build per-task evidence prompt ────────────────────────────────
        if not llm_tasks:
            # Nothing for LLM to review — return all MISSING
            results = [{
                'task_id':       t['task_id'],
                'status':        'MISSING',
                'reasoning':     'No relevant clauses found in contract.',
                'key_sentences': [],
            } for t in tasks]
            self._json_ok({'results': results})
            return

        _no_evidence = re.compile(
            r'does not (?:explicitly |clearly )?(?:mention|cover|include|specify|address)|'
            r'no mention|not mentioned|not specified|not covered|not explicitly|'
            r'cannot confirm|no evidence|unclear',
            re.IGNORECASE
        )

        # Sentences that are generic contract boilerplate — no real coverage signal
        _boilerplate_sent = re.compile(
            r'please tick|tick the box|service agreement will cover|'
            r'shall continue for\b|as follows\s*:|listed in appendix|the products listed|'
            r'see attached|subject to the following|terms and conditions|'
            r'^see\b|appendix\s+[a-z]\b|refer to|as per|schedule\s+\d',
            re.IGNORECASE
        )

        def _filter_evidence(sents):
            """Remove boilerplate sentences; fall back gracefully if all are filtered."""
            cleaned = [s for s in sents
                       if len(s.strip()) >= 30 and not _boilerplate_sent.search(s)]
            return cleaned if cleaned else [s for s in sents if len(s.strip()) >= 20]

        def _build_prompt(chunk):
            blocks = []
            for i, task in enumerate(chunk):
                evidence = _filter_evidence(task_top_sentences[task['task_id']])
                ev_lines = '\n   '.join(f'"{s[:220]}"' for s in evidence)
                action_hint = task.get('task_action', '')[:300].strip()
                task_desc = f"{task['task_title']}"
                if action_hint:
                    task_desc += f"\n   Task requires: {action_hint}"
                blocks.append(f"{i+1}. {task_desc}\n   Evidence: {ev_lines}")
            return (
                "You are a PPM compliance reviewer. Answer each task independently.\n"
                "COVERED = the evidence contains specific words or phrases that directly refer to THIS task (not just general 'service visits' or 'inspections' language).\n"
                "PARTIAL = evidence mentions related equipment/work but does not specifically name THIS task.\n"
                "MISSING = task not mentioned by name, OR evidence is only generic (visit schedules, access clauses, payment terms, scope headers). Default to MISSING when unsure.\n"
                "Do NOT infer that a general service visit covers a specific task unless the task is named.\n"
                "Reply with EXACTLY one line per task using its task number. Format:\n"
                "  1. COVERED: \"exact sentence from contract\"\n"
                "  1. PARTIAL: \"exact sentence from contract\"\n"
                "  1. MISSING: not found\n"
                "The number is the task's position in the list below (starting at 1).\n"
                "For COVERED or PARTIAL, quote one sentence verbatim in double quotes. Do not paraphrase.\n\n"
                + '\n'.join(blocks)
            )

        def _parse_chunk(raw_text, chunk, offset):
            out = {}
            for i, task in enumerate(chunk):
                pattern = re.compile(
                    rf'^\s*{i+1}[\.\)]\s*(COVERED|PARTIAL|MISSING)(?:[:\s\-]+(.+))?',
                    re.IGNORECASE | re.MULTILINE
                )
                m    = pattern.search(raw_text)
                line = ''
                if not m:
                    # Tier 1b: handle "STATUS: N. VERDICT: ..." or "VERDICT: N. ..." prefix
                    pattern2 = re.compile(
                        rf'^\s*(?:\w+:\s*)?{i+1}[\.\)]\s*(COVERED|PARTIAL|MISSING)(?:[:\s\-]+(.+))?',
                        re.IGNORECASE | re.MULTILINE
                    )
                    m = pattern2.search(raw_text)
                if not m:
                    # Tier 1c: bare "VERDICT" or "VERDICT: ..." at line start (no number)
                    for line in raw_text.split('\n'):
                        m2 = re.match(
                            r'^\s*(COVERED|PARTIAL|MISSING)(?:[:/\s\-]+(.+))?',
                            line, re.IGNORECASE
                        )
                        if m2:
                            m = m2
                            break
                if not m:
                    # Tier 2: any line containing a verdict word that also mentions task title words
                    title_words = set(task['task_title'].lower().split())
                    for line in raw_text.split('\n'):
                        vm = re.search(r'\b(COVERED|PARTIAL|MISSING)\b', line, re.IGNORECASE)
                        if vm:
                            line_words = set(line.lower().split())
                            if title_words & line_words:
                                m = vm
                                break
                if m:
                    status    = m.group(1).upper() if m.lastindex and m.lastindex >= 1 else m.group(0).upper()
                    reasoning = (m.group(2).strip() if m.lastindex and m.lastindex >= 2 and m.group(2)
                                 else line.replace(m.group(0), '').strip())
                    if status == 'PARTIAL' and _no_evidence.search(reasoning):
                        status = 'MISSING'
                if not m and len(chunk) == 1:
                    # Tier 3: single-task chunk — accept any line with a verdict word, any number
                    for line in raw_text.split('\n'):
                        m2 = re.search(r'\b(COVERED|PARTIAL|MISSING)\b', line, re.IGNORECASE)
                        if m2:
                            m = m2
                            break
                if not m:
                    status    = 'MISSING'
                    reasoning = 'Could not parse LLM response for this task.'
                citation_m = _citation_re.search(reasoning) if status != 'MISSING' else None
                citation   = _citation_noise.sub('', citation_m.group(1)).strip(' \u2022\u00b7\u2013\u2014-"\u201c\u201d') if citation_m else None
                if citation and len(citation) < 30:
                    citation = None
                out[task['task_id']] = {
                    'task_id':       task['task_id'],
                    'status':        status,
                    'reasoning':     reasoning,
                    'citation':      citation,
                    'key_sentences': _filter_evidence(task_top_sentences.get(task['task_id'], [])),
                }
            return out

        # ── 4. Process in chunks of 2 — faster per-call for larger models ──
        CHUNK_SIZE = 2
        llm_results = {}
        for chunk_start in range(0, len(llm_tasks), CHUNK_SIZE):
            chunk  = llm_tasks[chunk_start:chunk_start + CHUNK_SIZE]
            prompt = _build_prompt(chunk)
            print(f"[llm_check_batch] chunk {chunk_start//CHUNK_SIZE + 1}: "
                  f"{len(chunk)} tasks, prompt len={len(prompt)} chars")
            try:
                if _llm is not None:
                    # ── llama.cpp path ────────────────────────────────────────
                    max_tok = len(chunk) * 100 + 20
                    output  = _llm.create_chat_completion(
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=max_tok,
                        temperature=0.1,
                    )
                    raw = output['choices'][0]['message']['content'].strip()
                else:
                    # ── Ollama fallback (chat API — handles instruct template correctly) ─
                    import urllib.request as _ur
                    body   = json.dumps({
                        'model': chat_model,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'stream': False,
                        'options': {'temperature': 0.1,
                                    'num_predict': len(chunk) * 100 + 20,
                                    'num_ctx': 4096},
                    }).encode()
                    req      = _ur.Request('http://localhost:11434/api/chat',
                                           data=body, headers={'Content-Type': 'application/json'})
                    llm_resp = json.loads(_ur.urlopen(req, timeout=300).read())
                    raw      = llm_resp.get('message', {}).get('content', '').strip()
                print(f"[llm_check_batch] chunk {chunk_start//CHUNK_SIZE + 1} raw: {raw[:200]!r}")
                llm_results.update(_parse_chunk(raw, chunk, chunk_start))
            except Exception as exc:
                print(f"[llm_check_batch] chunk {chunk_start//CHUNK_SIZE + 1} ERROR: {exc}")
                # Fall back: mark all tasks in this chunk as MISSING
                for task in chunk:
                    llm_results[task['task_id']] = {
                        'task_id':       task['task_id'],
                        'status':        'MISSING',
                        'reasoning':     f'LLM error: {exc}',
                        'key_sentences': task_top_sentences.get(task['task_id'], []),
                    }

        # Merge LLM results with pre-decided MISSING tasks, preserving original order
        results = []
        for task in tasks:
            if task['task_id'] in llm_results:
                results.append(llm_results[task['task_id']])
            else:
                results.append({
                    'task_id':       task['task_id'],
                    'status':        'MISSING',
                    'reasoning':     'No relevant clauses found in contract.',
                    'key_sentences': [],
                })

        self._json_ok({'results': results})

    def _llm_summary(self, payload):
        total       = payload.get('total', 0)
        covered     = payload.get('covered', 0)
        partial     = payload.get('partial', 0)
        missing     = payload.get('missing', 0)
        high_miss   = payload.get('high_missing', 0)
        high_total  = payload.get('high_total', 0)
        top_gaps    = payload.get('top_gaps', [])   # [{ref, label, missing_high}]
        chat_model  = payload.get('model', 'phi3:mini')

        if total == 0 or not top_gaps:
            self._json_error(422, 'Insufficient data for summary')
            return

        cov_pct   = round((covered + partial) / total * 100)
        gaps_text = '\n'.join(
            f"  - {g['ref']}: {g.get('label', g['ref'])} ({g['missing_high']} HIGH missing)"
            for g in top_gaps[:3]
        )
        prompt = (
            "You are reviewing an NHS hospital PPM compliance assessment.\n\n"
            f"Data:\n"
            f"- {total} tasks assessed, {cov_pct}% coverage ({covered} covered, {partial} partial, {missing} missing)\n"
            f"- HIGH criticality gaps: {high_miss} of {high_total} HIGH tasks are MISSING\n"
            f"- Schedules with critical gaps:\n{gaps_text}\n\n"
            "Respond with EXACTLY these two lines and nothing else:\n"
            "Gap: [the most critical unaddressed obligation — name the specific schedule and what maintenance work is absent]\n"
            "Action: [one specific clause or addition to request from the contractor to close this gap]"
        )

        try:
            if _llm is not None:
                output = _llm.create_chat_completion(
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=120,
                    temperature=0.2,
                )
                raw = output['choices'][0]['message']['content'].strip()
            else:
                import urllib.request as _ur
                body = json.dumps({
                    'model': chat_model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'stream': False,
                    'options': {'temperature': 0.2, 'num_predict': 120, 'num_ctx': 1024},
                }).encode()
                req      = _ur.Request('http://localhost:11434/api/chat',
                                       data=body, headers={'Content-Type': 'application/json'})
                llm_resp = json.loads(_ur.urlopen(req, timeout=120).read())
                raw      = llm_resp.get('message', {}).get('content', '').strip()

            # Extract Gap: and Action: lines — ignore anything else the model adds
            lines  = [l.strip() for l in raw.split('\n') if l.strip()]
            gap    = next((l for l in lines if re.match(r'^gap\s*:', l, re.IGNORECASE)), None)
            action = next((l for l in lines if re.match(r'^action\s*:', l, re.IGNORECASE)), None)
            if not gap or not action:
                self._json_error(422, f'LLM summary unparseable: {raw[:200]}')
                return
            summary = gap + '\n' + action
            print(f'[llm_summary] {summary!r}')
            self._json_ok({'summary': summary})
        except Exception as exc:
            self._json_error(500, f'LLM summary failed: {exc}')

    def _json_ok(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, message):
        body = json.dumps({'error': message}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise

os.chdir(_BUNDLE_DIR)

def _find_free_port(start=8000, attempts=20):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free port found in range %d–%d' % (start, start + attempts))

PORT = _find_free_port()
httpd = http.server.HTTPServer(('', PORT), PPMHandler)
url = f'http://localhost:{PORT}/PPM_Compliance_Checker.html'
print(f'Open: {url}')

try:
    import webview
    import threading
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    window = webview.create_window(
        'PPM Compliance Checker', url,
        width=1440, height=900, min_size=(900, 600)
    )
    webview.start()
except ImportError:
    # pywebview not installed — fall back to browser
    webbrowser.open(url)
    httpd.serve_forever()
