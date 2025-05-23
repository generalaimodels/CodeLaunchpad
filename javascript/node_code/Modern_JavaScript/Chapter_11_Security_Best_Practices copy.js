/**************************************************************************************************
 * Chapter 11 | Security Best Practices
 * -----------------------------------------------------------------------------------------------
 * One self‑contained .js playground; 5 domains × ≥5 runnable / illustrative examples each.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION XSS — Cross‑Site Scripting & CSRF Mitigation            */
/*───────────────────────────────────────────────────────────────────*/

/* XSS‑Example‑1:  Escaping user input for innerHTML */
(function () {
    const escape = s => s.replace(/[&<>"'`=\/]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;', '/':'&#x2F;', '`':'&#x60;','=':'&#x3D;' }[c]));
    const user  = '<img src=x onerror=alert(1)>';
    document.body.innerHTML = `<p>${escape(user)}</p>`;
  })();
  
  /* XSS‑Example‑2:  Using textContent instead of innerHTML */
  (function () {
    const span = document.createElement('span');
    span.textContent = '<b>safe</b>';
    document.body.appendChild(span);
  })();
  
  /* XSS‑Example‑3:  HTTP‑only & SameSite cookies for CSRF */
  (function () {
    document.cookie = 'sid=abc123; HttpOnly; Secure; SameSite=Lax';
    console.log('XSS‑3 cookie set with SameSite');
  })();
  
  /* XSS‑Example‑4:  CSRF token pattern */
  (function () {
    const csrf = crypto.randomUUID();
    sessionStorage.setItem('csrf', csrf);
    const form = document.createElement('form'); form.method = 'POST';
    const input = document.createElement('input'); input.type='hidden'; input.name='csrf'; input.value=csrf;
    form.appendChild(input); document.body.appendChild(form);
  })();
  
  /* XSS‑Example‑5:  Double submit cookie verification */
  (function () {
    const token = crypto.randomUUID();
    document.cookie = `csrf=${token}; SameSite=Strict`;
    const reqHeaders = { 'X-CSRF-Token': token };
    console.log('XSS‑5 header matches cookie?', reqHeaders['X-CSRF-Token'] === token);
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION CSP — Content Security Policy                           */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* CSP‑Example‑1:  Meta tag policy */
  (function () {
    const meta = document.createElement('meta');
    meta.httpEquiv = 'Content-Security-Policy';
    meta.content = "default-src 'self'";
    document.head.appendChild(meta);
  })();
  
  /* CSP‑Example‑2:  Script nonce usage */
  (function () {
    const nonce = btoa(crypto.getRandomValues(new Uint8Array(8)).join(''));
    document.head.appendChild(Object.assign(document.createElement('meta'), {
      httpEquiv:'Content-Security-Policy',
      content:`script-src 'self' 'nonce-${nonce}'`
    }));
    const scr = document.createElement('script');
    scr.textContent = 'console.log("CSP‑2 inline allowed");';
    scr.setAttribute('nonce', nonce);
    document.body.appendChild(scr);
  })();
  
  /* CSP‑Example‑3:  Report‑only header sample object */
  const cspReportOnly = {
    'Content-Security-Policy-Report-Only': "img-src 'self'; report-uri /csp-report"
  };
  console.log('CSP‑3 report‑only header:', cspReportOnly);
  
  /* CSP‑Example‑4:  Trusted Types enforcement */
  (async () => {
    if ('trustedTypes' in window) {
      window.trustedTypes.createPolicy('default', { createHTML: s => s }); // demo policy
      try {
        document.body.innerHTML = '<img src=x>';
      } catch (e) { console.log('CSP‑4 TrustedTypes blocked unsafe HTML'); }
    }
  })();
  
  /* CSP‑Example‑5:  Subresource Integrity (SRI) */
  (function () {
    const script = document.createElement('script');
    script.src = 'https://cdn.example.com/lib.js';
    script.integrity = 'sha384-abcdef123456';
    script.crossOrigin = 'anonymous';
    console.log('CSP‑5 SRI attributes set');
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION AUTH — Secure Authentication (OAuth2, JWT)              */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* AUTH‑Example‑1:  Generating JWT (HS256) */
  (async () => {
    const { createHmac } = await import('https://esm.run/@eyna/crypto?bundle');
    const header = btoa(JSON.stringify({ alg:'HS256', typ:'JWT' }));
    const payload = btoa(JSON.stringify({ sub:'123', exp:Date.now()/1000+3600 }));
    const sig = createHmac('sha256','secret').update(`${header}.${payload}`).digest('base64url');
    const jwt = `${header}.${payload}.${sig}`;
    console.log('AUTH‑1 JWT:', jwt.split('.').length === 3);
  })();
  
  /* AUTH‑Example‑2:  Verify JWT expiry */
  (function () {
    const decode = t => JSON.parse(atob(t.split('.')[1]));
    const expired = decode('eyJhbGciOi...'); // placeholder
    // console.log('AUTH‑2 expired?', expired.exp < Date.now()/1000);
  })();
  
  /* AUTH‑Example‑3:  OAuth2 PKCE code verifier generation */
  (function () {
    const verifier = btoa(crypto.getRandomValues(new Uint8Array(32)).join(''));
    console.log('AUTH‑3 PKCE verifier length:', verifier.length);
  })();
  
  /* AUTH‑Example‑4:  State param against CSRF */
  (function () {
    const state = crypto.randomUUID();
    sessionStorage.setItem('oauth_state', state);
    const url = `https://auth?state=${state}`;
    console.log('AUTH‑4 auth URL:', url.includes(state));
  })();
  
  /* AUTH‑Example‑5:  Refresh token rotation model */
  const refreshFlow = {
    step1: 'client sends refresh token',
    step2: 'server issues new access & refresh tokens',
    step3: 'old refresh token invalidated'
  };
  console.log('AUTH‑5 rotation flow keys:', Object.keys(refreshFlow).length);
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION SAN — Sanitization & Input Validation                   */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* SAN‑Example‑1:  Joi schema validation */
  try {
    const Joi = (await import('https://esm.run/joi?bundle')).default;
    const schema = Joi.object({ id:Joi.number().integer().positive().required() });
    console.log('SAN‑1 valid id?', !schema.validate({ id:5 }).error);
  } catch {}
  
  /* SAN‑Example‑2:  HTML sanitization using DOMPurify */
  (async () => {
    const DOMPurify = (await import('https://esm.run/dompurify?bundle')).default;
    const dirty = '<img src=x onerror=alert(1)>';
    console.log('SAN‑2 clean:', DOMPurify.sanitize(dirty));
  })();
  
  /* SAN‑Example‑3:  RegExp email validation */
  (function () {
    const emailRE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    console.log('SAN‑3 email ok?', emailRE.test('a@b.com'));
  })();
  
  /* SAN‑Example‑4:  SQL parameterized query stub */
  (function () {
    const query = 'SELECT * FROM users WHERE id = ?';
    const params = [42];
    console.log('SAN‑4 paramized query tokens:', params.length);
  })();
  
  /* SAN‑Example‑5:  Path traversal prevention */
  (function () {
    const path = '../etc/passwd';
    const safe = require('path').normalize(path).replace(/^(\.\.(\/|\\|$))+/, '');
    console.log('SAN‑5 sanitized path:', safe);
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION DEP — Dependency Vulnerability Scanning                 */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* DEP‑Example‑1:  npm audit programmatic */
  (async () => {
    const { execSync } = require('child_process');
    console.log('DEP‑1 audit json keys:',
      Object.keys(JSON.parse(execSync('npm audit --json').toString())));
  })();
  
  /* DEP‑Example‑2:  Snyk test CLI snippet */
  const snykCmd = 'snyk test --severity-threshold=high';
  console.log('DEP‑2 cmd contains snyk?', snykCmd.includes('snyk'));
  
  /* DEP‑Example‑3:  GitHub Dependabot config skeleton */
  const dependabot = {
    version: 2,
    updates: [{ packageEcosystem:'npm', directory:'/', schedule:{ interval:'daily' } }]
  };
  console.log('DEP‑3 dependabot interval:', dependabot.updates[0].schedule.interval);
  
  /* DEP‑Example‑4:  Semgrep security scan */
  const semgrep = 'semgrep --config=p/owasp-top-ten';
  console.log('DEP‑4 semgrep cmd length:', semgrep.length);
  
  /* DEP‑Example‑5:  CVSS score threshold enforcement */
  (function () {
    const vulns = [{ id:'X', cvss:9.8 }, { id:'Y', cvss:4.3 }];
    const high = vulns.filter(v => v.cvss >= 7);
    console.log('DEP‑5 high severity count:', high.length);
  })();