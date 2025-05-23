/****************************************************************************************
 * Chapter 10 | Testing, Debugging & QA
 ****************************************************************************************/

/* SECTION UT — Unit Testing (Jest, Mocha, AVA) */

/* UT-1: Jest */
test('add() should return sum', () => {
    function add(a, b) { return a + b; }
    expect(add(2, 3)).toBe(5);
  });
  
  /* UT-2: Mocha + Chai */
  const { expect: chaiExpect } = require('chai');
  describe('multiply()', () => {
    function multiply(a, b) { return a * b; }
    it('multiplies numbers', () => {
      chaiExpect(multiply(3, 4)).to.equal(12);
    });
  });
  
  /* UT-3: AVA */
  import test from 'ava';
  function subtract(a, b) { return a - b; }
  test('subtract()', t => {
    t.is(subtract(5, 3), 2);
  });
  
  /* UT-4: Tape */
  const tape = require('tape');
  tape('divide()', t => {
    function divide(a, b) { t.plan(1); t.equal(a / b, 2); }
    divide(6, 3);
  });
  
  /* UT-5: Node assert */
  const assert = require('assert');
  function mod(a, b) { return a % b; }
  assert.strictEqual(mod(7, 3), 1);
  
  
  /* SECTION IT — Integration & E2E (Cypress, Playwright) */
  
  /* IT-1: Cypress */
  describe('Cypress E2E', () => {
    it('visits example.com', () => {
      cy.visit('https://example.com');
      cy.get('h1').contains('Example Domain');
    });
  });
  
  /* IT-2: Playwright */
  import { test as pwTest, expect as pwExpect } from '@playwright/test';
  pwTest('homepage title', async ({ page }) => {
    await page.goto('https://example.com');
    pwExpect(await page.title()).toBe('Example Domain');
  });
  
  /* IT-3: Puppeteer */
  const puppeteer = require('puppeteer');
  (async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto('https://example.com');
    console.log(await page.$eval('h1', el => el.textContent));
    await browser.close();
  })();
  
  /* IT-4: WebdriverIO */
  const { remote } = require('webdriverio');
  (async () => {
    const browser = await remote({ capabilities: { browserName: 'chrome' } });
    await browser.url('https://example.com');
    console.log(await (await browser.$('h1')).getText());
    await browser.deleteSession();
  })();
  
  /* IT-5: SuperTest */
  const request = require('supertest');
  const express = require('express');
  const app = express();
  app.get('/ping', (req, res) => res.json({ pong: true }));
  request(app)
    .get('/ping')
    .expect('Content-Type', /json/)
    .expect(200, { pong: true });
  
  
  /* SECTION TDD — TDD / BDD Workflows */
  
  /* TDD-1: Jest Red‑Green */
  test('reverse() returns reversed string', () => {
    expect(reverse('abc')).toBe('cba');
  });
  function reverse(s) { return s.split('').reverse().join(''); }
  
  /* TDD-2: Mocha BDD */
  describe('isPalindrome()', () => {
    it('detects palindrome', () => {
      chaiExpect(isPalindrome('madam')).to.be.true;
    });
  });
  function isPalindrome(s) { return s === s.split('').reverse().join(''); }
  
  /* TDD-3: AVA hooks */
  test.before(t => { t.context.data = [1, 2, 3]; });
  test('array length', t => { t.is(t.context.data.length, 3); });
  test.after(() => { /* cleanup */ });
  
  /* TDD-4: GitHub‑driven feature branch */
  /*
  1. Create issue → 2. Branch feature/... → 3. Write failing tests → 4. Implement → 5. PR & Merge
  */
  
  /* TDD-5: Cucumber.js */
  const { Given, When, Then } = require('@cucumber/cucumber');
  Given('user on login page', () => { /* nav */ });
  When('enters valid credentials', () => { /* fill */ });
  Then('sees dashboard', () => { /* assert */ });
  
  
  /* SECTION MCK — Mocking & Stubbing */
  
  /* MCK-1: Jest.mock */
  jest.mock('axios');
  import axios from 'axios';
  axios.get.mockResolvedValue({ data: { id: 1 } });
  test('fetchData()', async () => {
    const res = await axios.get('/url');
    expect(res.data.id).toBe(1);
  });
  
  /* MCK-2: Sinon.stub */
  const sinon = require('sinon');
  const fsStub = sinon.stub(require('fs'), 'readFileSync').returns('stubbed');
  console.log(require('fs').readFileSync('file'));
  
  /* MCK-3: Proxyquire */
  const proxyquire = require('proxyquire');
  const mod = proxyquire('./mod', { './dep': { foo: () => 42 } });
  console.log(mod.foo());
  
  /* MCK-4: Nock */
  const nock = require('nock');
  nock('https://api.com').get('/data').reply(200, { ok: true });
  const api = require('./api');
  api.getData().then(console.log);
  
  /* MCK-5: Manual DI */
  function sendEmail(service, msg) { return service.send(msg); }
  const stubEmail = { send: m => `stub:${m}` };
  console.log(sendEmail(stubEmail, 'hi'));
  
  
  /* SECTION COV — Coverage & Quality Metrics */
  
  /* COV-1: Istanbul Instrument API */
  import { createInstrumenter } from 'istanbul-lib-instrument';
  const inst = createInstrumenter();
  const code = 'function add(a,b){return a+b;}';
  console.log(inst.instrumentSync(code, 'file.js'));
  
  /* COV-2: nyc in package.json */
  /*
  "scripts": { "test":"nyc mocha" },
  "nyc": { "reporter":["text","lcov"], "exclude":["test/fixtures"] }
  */
  
  /* COV-3: Jest thresholds */
  const jestConfig = {
    coverageThreshold: {
      global: { branches: 80, functions: 80, lines: 80, statements: 80 }
    }
  };
  console.log(jestConfig);
  
  /* COV-4: Generate report */
  // npx nyc report --reporter=html
  
  /* COV-5: Coverage badge */
  // ![coverage](https://img.shields.io/badge/coverage-85%25-green)
  
  
  /* SECTION DBG — Debugger, Breakpoints & Source Maps */
  
  /* DBG-1: debugger; */
  function complexCalc(x) { debugger; return x * 2; }
  complexCalc(5);
  
  /* DBG-2: Node inspect */
  // node --inspect-brk chapter10.js
  // open chrome://inspect
  
  /* DBG-3: DevTools breakpoints */
  // In DevTools → Sources → click line number to set breakpoint
  
  /* DBG-4: Babel source‑maps */
  // .babelrc: { "sourceMaps":"inline" }
  
  /* DBG-5: VSCode launch.json */
  const launchJson = {
    version: "0.2.0",
    configurations: [{
      type: "node",
      request: "launch",
      name: "Launch Program",
      program: "${workspaceFolder}/chapter10.js",
      sourceMaps: true
    }]
  };
  console.log(launchJson);