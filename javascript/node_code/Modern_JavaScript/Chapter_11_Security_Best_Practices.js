/****************************************************************************************
 * Chapter 11 | Security Best Practices
 * Single-file .js with ≥5 examples per topic; no extra content.
 ****************************************************************************************/

/* SECTION 1: XSS & CSRF MITIGATION */

/* 1.1: Safe insertion using textContent (prevents XSS) */
(function xss1() {
    const userInput = '<img src=x onerror=alert(1)>'
    const el = document.createElement('div')
    el.textContent = userInput              // safe: renders literal string
    document.body.appendChild(el)
  })()
  
  /* 1.2: Sanitizing HTML with DOMPurify */
  (function xss2() {
    // assume DOMPurify loaded
    const dirty = '<svg><script>alert(2)</script></svg>'
    const clean = DOMPurify.sanitize(dirty) // strips <script>
    document.body.innerHTML = clean
  })()
  
  /* 1.3: HTTP-only, SameSite cookies to mitigate CSRF */
  (function csrf1() {
    // server sets:
    // Set-Cookie: session=abc123; HttpOnly; SameSite=Strict; Secure
    // browser will not send cookie on cross-site POST
  })()
  
  /* 1.4: Synchronizer token pattern for CSRF */
  (function csrf2() {
    // server side (Node/Express pseudo):
    // app.get('/form', (req,res)=> {
    //   req.session.csrf = crypto.randomBytes(16).toString('hex')
    //   res.send(`<form><input type="hidden" name="csrf" value="${req.session.csrf}">…`)
    // })
    // app.post('/submit', (req,res)=>{
    //   if (req.body.csrf !== req.session.csrf) throw Error('CSRF mismatch')
    // })
  })()
  
  /* 1.5: Double-submit cookie pattern */
  (function csrf3() {
    // client sends token both in cookie and header
    fetch('/submit', {
      method:'POST',
      credentials:'include',
      headers:{ 'X-CSRF-Token': document.cookie.match(/csrf=([^;]+)/)[1] },
      body: JSON.stringify({ data:123 })
    })
  })()
  
  /* SECTION 2: CONTENT SECURITY POLICY (CSP) */
  
  /* 2.1: Meta tag CSP (blocks inline scripts) */
  (function csp1() {
    document.head.innerHTML +=
      `<meta http-equiv="Content-Security-Policy"
        content="default-src 'self'; script-src 'self'; object-src 'none'">`
  })()
  
  /* 2.2: Express helmet to set CSP header */
  (function csp2() {
    // const express = require('express'), helmet = require('helmet')
    // const app = express()
    // app.use(helmet.contentSecurityPolicy({ directives:{
    //   defaultSrc:["'self'"], scriptSrc:["'self'","https://cdn.example.com"]
    // }}))
  })()
  
  /* 2.3: nonce-based CSP for trusted inline scripts */
  (function csp3() {
    const nonce = btoa(crypto.getRandomValues(new Uint8Array(16)))
    document.head.innerHTML +=
      `<meta http-equiv="Content-Security-Policy"
        content="script-src 'nonce-${nonce}';">`
    const script = document.createElement('script')
    script.nonce = nonce
    script.textContent = "console.log('trusted inline')"
    document.head.appendChild(script)
  })()
  
  /* 2.4: report-only mode */
  (function csp4() {
    // <meta http-equiv="Content-Security-Policy-Report-Only"
    //   content="default-src 'self'; report-uri /csp-report">
  })()
  
  /* 2.5: blocking mixed content */
  (function csp5() {
    // app.use(helmet.contentSecurityPolicy({ directives:{
    //   upgradeInsecureRequests: []
    // }}))
  })()
  
  /* SECTION 3: SECURE AUTHENTICATION (OAuth2, JWT) */
  
  /* 3.1: Generating JWT with HS256 */
  (function jwt1() {
    // const jwt = require('jsonwebtoken')
    // const token = jwt.sign({ sub: 'user1' }, 'secretKey', { algorithm:'HS256', expiresIn:'1h' })
    // console.log(token)
  })()
  
  /* 3.2: Verifying JWT and handling exceptions */
  (function jwt2() {
    // try {
    //   const payload = jwt.verify(token, 'secretKey')
    // } catch (e) {
    //   if (e.name==='TokenExpiredError') handleExpired()
    //   else handleInvalid()
    // }
  })()
  
  /* 3.3: OAuth2 Authorization Code flow skeleton */
  (function oauth2() {
    // Redirect user:
    // res.redirect(`https://auth.example.com/authorize?response_type=code&client_id=ID&redirect_uri=${uri}`)
    // Exchange code:
    // const resp = await fetch('https://auth.example.com/token', {
    //   method:'POST', body:new URLSearchParams({ code, client_id, client_secret, grant_type:'authorization_code' })
    // })
    // const { access_token } = await resp.json()
  })()
  
  /* 3.4: Refresh token flow */
  (function refresh() {
    // const refreshToken = req.cookies.refresh
    // const resp = await fetch('https://auth.example.com/token', {
    //   method:'POST', body:new URLSearchParams({ grant_type:'refresh_token', refresh_token:refreshToken })
    // })
    // const { access_token, refresh_token } = await resp.json()
  })()
  
  /* 3.5: Store JWT in HttpOnly cookie (mitigates XSS theft) */
  (function jwt3() {
    // res.cookie('jwt', token, { httpOnly:true, secure:true, sameSite:'Strict' })
  })()
  
  /* SECTION 4: SANITIZATION & INPUT VALIDATION */
  
  /* 4.1: Validator.js for email & URL */
  (function val1() {
    // const { isEmail, isURL } = require('validator')
    // isEmail('foo@bar.com') // true
    // isURL('https://example.com') // true
  })()
  
  /* 4.2: Joi schema validation */
  (function val2() {
    // const Joi = require('joi')
    // const schema = Joi.object({ user:Joi.string().alphanum().min(3).required(), age:Joi.number().integer().min(0) })
    // const { error, value } = schema.validate({ user:'bob', age:30 })
    // if (error) throw error
  })()
  
  /* 4.3: SQL parameterized queries (prevents injection) */
  (function val3() {
    // const mysql = require('mysql2/promise')
    // const conn = await mysql.createConnection({...})
    // const [rows] = await conn.execute('SELECT * FROM users WHERE id = ?', [userId])
  })()
  
  /* 4.4: DOMPurify to sanitize rich‑text input */
  (function val4() {
    // const clean = DOMPurify.sanitize(userPostedHTML, { ALLOWED_TAGS:['b','i','a'], ALLOWED_ATTR:['href'] })
  })()
  
  /* 4.5: Custom whitelist sanitizer */
  (function val5() {
    const whitelist = { name:/^[A-Za-z ]{1,50}$/, age:/^[0-9]{1,3}$/ }
    function sanitize(input, field) {
      const re = whitelist[field]
      if (!re) throw Error('No rules')
      if (!re.test(input)) throw Error('Invalid '+field)
      return input
    }
    sanitize('Alice','name')  // OK
    // sanitize('Alice<script>','name') // throws
  })()
  
  /* SECTION 5: DEPENDENCY VULNERABILITY SCANNING */
  
  /* 5.1: npm audit via child_process */
  (function audit1() {
    const { exec } = require('child_process')
    exec('npm audit --json', (err, out) => {
      if (err) console.error(err)
      else console.log('Vulns:', JSON.parse(out).metadata.vulnerabilities)
    })
  })()
  
  /* 5.2: Snyk CLI scan */
  (function audit2() {
    // exec('snyk test --json', (e,o) => {
    //   const res = JSON.parse(o)
    //   console.log('Snyk high:', res.vulnerabilities.filter(v=>v.severity==='high'))
    // })
  })()
  
  /* 5.3: Programmatic audit with npm-audit */
  (function audit3() {
    // const audit = require('npm-audit') // hypothetical
    // audit({ json:true }).then(report=> console.log(report.advisories))
  })()
  
  /* 5.4: Checking outdated deps */
  (function audit4() {
    const { exec } = require('child_process')
    exec('npm outdated --json', (e,o)=> {
      if (!e) console.log('Outdated:', JSON.parse(o))
    })
  })()
  
  /* 5.5: CI integration: fail on vulnerabilities */
  (function audit5() {
    // in GitHub Actions:
    // - run: npm audit --audit-level=moderate
    //   continue-on-error: false
  })()