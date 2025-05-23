/**************************************************************************************************
 * Chapter 10 | Testing, Debugging & QA
 * =========================================================================
 * Single .js playground file — copy/paste into IDE or Node (≥16).  
 * 6 domains × ≥5 concise, illustrative examples each.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION UNIT — Unit Testing (Jest, Mocha, AVA)                  */
/*───────────────────────────────────────────────────────────────────*/

/* UNIT‑Example‑1:  Jest test + assertion */
const jestTest = `
test('adds', () => {
  expect(1 + 2).toBe(3);
});
`;
console.log('UNIT‑1 Jest snippet length:', jestTest.length);

/* UNIT‑Example‑2:  Mocha with Chai expect */
const mochaTest = `
const { expect } = require('chai');
describe('math', () => {
  it('multiplies', () => expect(2 * 3).to.equal(6));
});
`;
console.log('UNIT‑2 Mocha snippet lines:', mochaTest.trim().split('\n').length);

/* UNIT‑Example‑3:  AVA macro test */
const avaTest = `
import test from 'ava';
const macro = (t, a, b, s) => t.is(a + b, s);
macro.title = (provided, a, b, s) => \`\${a}+\${b}=\${s}\`;
test(macro, 2, 3, 5);
`;
console.log('UNIT‑3 AVA characters:', avaTest.length);

/* UNIT‑Example‑4:  Table‑driven test (Jest.each) */
const table = `
it.each([
  [1, 1, 2],
  [2, 3, 5]
])('%i + %i = %i', (a, b, sum) => {
  expect(a + b).toBe(sum);
});
`;
console.log('UNIT‑4 table rows:', (table.match(/\n/g) || []).length);

/* UNIT‑Example‑5:  Parameterized Mocha using Array.forEach */
const paramMocha = `
[ [4,2], [6,3] ].forEach(([n, d]) => {
  it(\`\${n}/\${d}\`, () => expect(n / d).to.equal(2));
});
`;
console.log('UNIT‑5 param:', paramMocha.includes('forEach'));

/*───────────────────────────────────────────────────────────────────*/
/* SECTION INT — Integration & E2E (Cypress, Playwright)           */
/*───────────────────────────────────────────────────────────────────*/

/* INT‑Example‑1:  Cypress visit + assertion */
const cypress = `
describe('home', () => {
  it('loads', () => {
    cy.visit('/');
    cy.contains('Welcome');
  });
});
`;
console.log('INT‑1 Cypress lines:', cypress.trim().split('\n').length);

/* INT‑Example‑2:  Playwright page interaction */
const pw = `
import { test, expect } from '@playwright/test';
test('login', async ({ page }) => {
  await page.goto('/');
  await page.fill('#user', 'admin');
  await page.fill('#pass', 'secret');
  await page.click('text=Login');
  await expect(page).toHaveURL('/dashboard');
});
`;
console.log('INT‑2 Playwright length:', pw.length);

/* INT‑Example‑3:  REST integration test with supertest */
const restTest = `
const req = require('supertest')('http://localhost:3000');
describe('GET /health', () => {
  it('200 OK', async () => {
    await req.get('/health').expect(200);
  });
});
`;
console.log('INT‑3 supertest contains GET?', restTest.includes('GET'));

/* INT‑Example‑4:  Docker‑compose up before tests (shell) */
const composeCmd = 'docker compose -f docker-test.yml up -d';
console.log('INT‑4 compose cmd:', composeCmd);

/* INT‑Example‑5:  Smoke test script */
const smoke = `
node server.js &
sleep 2
curl -f localhost:3000/health
`;
console.log('INT‑5 smoke newlines:', smoke.split('\n').length);

/*───────────────────────────────────────────────────────────────────*/
/* SECTION TDD — TDD / BDD Workflows                               */
/*───────────────────────────────────────────────────────────────────*/

/* TDD‑Example‑1:  Red‑Green‑Refactor checklist */
const cycle = ['RED', 'GREEN', 'REFACTOR'];
console.log('TDD‑1 cycle:', cycle.join(' → '));

/* TDD‑Example‑2:  Failing test first (Jest) */
const failing = `
test('fizzBuzz(3) returns "Fizz"', () => {
  expect(fizzBuzz(3)).toBe('Fizz'); // fizzBuzz not yet implemented
});
`;
console.log('TDD‑2 failing test chars:', failing.length);

/* TDD‑Example‑3:  BDD style with describe/it */
const bdd = `
describe('Stack', () => {
  context('when empty', () => {
    it('isEmpty returns true', () => {/* ... */});
  });
});
`;
console.log('TDD‑3 uses context?', bdd.includes('context'));

/* TDD‑Example‑4:  Given‑When‑Then comments */
const gwt = `
/* GIVEN a user */
/* WHEN they login */
/* THEN dashboard is shown */
`;
console.log('TDD‑4 GWT lines:', gwt.trim().split('\n').length);

/* TDD‑Example‑5:  Watch mode CLI flag */
const jestWatch = 'jest --watchAll';
console.log('TDD‑5 watch flag present?', jestWatch.includes('--watch'));

/*───────────────────────────────────────────────────────────────────*/
/* SECTION MOCK — Mocking & Stubbing Strategies                    */
/*───────────────────────────────────────────────────────────────────*/

/* MOCK‑Example‑1:  jest.fn() spy */
const spy = jest ? jest.fn() : () => {};
console.log('MOCK‑1 spy typeof:', typeof spy);

/* MOCK‑Example‑2:  Manual stub object */
const dbStub = { save: () => Promise.resolve(1) };
console.log('MOCK‑2 stub save exists?', 'save' in dbStub);

/* MOCK‑Example‑3:  Sinon fake timers */
const sinonTimer = `
const clock = sinon.useFakeTimers();
doSomethingLater();
clock.tick(1000);
`;
console.log('MOCK‑3 contains tick?', sinonTimer.includes('tick'));

/* MOCK‑Example‑4:  Module mock (jest.mock) */
const moduleMock = `
jest.mock('./api', () => ({ fetch: jest.fn(() => 42) }));
`;
console.log('MOCK‑4 mock keyword?', moduleMock.includes('jest.mock'));

/* MOCK‑Example‑5:  Testdouble stand‑in */
const tdUsage = `
const td = require('testdouble');
const http = td.object(['get']);
td.when(http.get('/')).thenResolve({ ok:true });
`;
console.log('MOCK‑5 td object?', tdUsage.includes('td.object'));

/*───────────────────────────────────────────────────────────────────*/
/* SECTION COV — Code Coverage & Quality Metrics                   */
/*───────────────────────────────────────────────────────────────────*/

/* COV‑Example‑1:  Istanbul/nyc config */
const nycCfg = { reporter: ['text', 'html'], statements: 90 };
console.log('COV‑1 nyc stmt target:', nycCfg.statements);

/* COV‑Example‑2:  Jest coverage CLI */
const jestCov = 'jest --coverage';
console.log('COV‑2 cmd includes coverage?', jestCov.includes('--coverage'));

/* COV‑Example‑3:  Badge URL generator */
const badge = branch => `https://img.shields.io/badge/coverage-${branch}%25-brightgreen`;
console.log('COV‑3 100% badge:', badge(100));

/* COV‑Example‑4:  Mutant testing mention (Stryker) */
const stryker = `
stryker run --mutation-range 10..20
`;
console.log('COV‑4 stryker lines:', stryker.trim().split('\n').length);

/* COV‑Example‑5:  SonarQube quality gate JSON */
const sonarGate = { coverage: 85, vulnerabilities: 0 };
console.log('COV‑5 sonar coverage:', sonarGate.coverage);

/*───────────────────────────────────────────────────────────────────*/
/* SECTION DBG — Debugger, Breakpoints & Source Maps               */
/*───────────────────────────────────────────────────────────────────*/

/* DBG‑Example‑1:  debugger statement */
function debugFunc() {
  const x = 5;
  debugger; // sets breakpoint
  return x * 2;
}
console.log('DBG‑1 call result:', debugFunc());

/* DBG‑Example‑2:  Node inspect flag */
const inspectCmd = 'node --inspect-brk app.js';
console.log('DBG‑2 inspect flag?', inspectCmd.includes('--inspect'));

/* DBG‑Example‑3:  Source map comment */
const srcMap = `//# sourceMappingURL=app.js.map`;
console.log('DBG‑3 has sourceMappingURL?', srcMap.includes('sourceMappingURL'));

/* DBG‑Example‑4:  Chrome DevTools snippet for breakpoint */
const chromeSnippet = `
window.addEventListener('click', () => {
  debugger;
});
`;
console.log('DBG‑4 snippet length:', chromeSnippet.length);

/* DBG‑Example‑5:  VSCode launch.json fragment */
const launch = {
  config: {
    type: 'node',
    request: 'attach',
    port: 9229,
    sourceMaps: true
  }
};
console.log('DBG‑5 launch attach port:', launch.config.port);