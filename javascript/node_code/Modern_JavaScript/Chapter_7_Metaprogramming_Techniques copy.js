// Chapter 7: Metaprogramming Techniques

// 1. Reflect API & Metadata
// --------------------------

// Example 1: Reflect.get and Reflect.set
const obj1 = { a: 1 };
Reflect.set(obj1, 'b', 2);
console.log(Reflect.get(obj1, 'b')); // 2

// Example 2: Reflect.defineProperty vs Object.defineProperty
const obj2 = {};
Reflect.defineProperty(obj2, 'x', { value: 10, writable: false });
try { obj2.x = 20 } catch(e) { console.log('Cannot write x'); }
console.log(obj2.x); // 10

// Example 3: Reflect.has and Reflect.deleteProperty
const obj3 = { key: 'val' };
console.log(Reflect.has(obj3, 'key')); // true
Reflect.deleteProperty(obj3, 'key');
console.log(Reflect.has(obj3, 'key')); // false

// Example 4: Reflect.apply to call functions
function sum(a, b){ return a + b; }
const result4 = Reflect.apply(sum, null, [5,7]);
console.log(result4); // 12

// Example 5: Metadata with reflect-metadata
require('reflect-metadata');
class MetaDemo {}
Reflect.defineMetadata('role', 'admin', MetaDemo);
const role = Reflect.getMetadata('role', MetaDemo);
console.log(role); // 'admin'


// 2. Proxy Objects & Traps
// -------------------------

// Example 1: Property access logging
const target1 = { msg: 'hello' };
const proxy1 = new Proxy(target1, {
  get(t, p){ console.log('get', p); return t[p]; }
});
proxy1.msg; // logs get msg

// Example 2: Validation proxy
const person = { age: 25 };
const proxy2 = new Proxy(person, {
  set(t, p, v){
    if(p === 'age' && (typeof v !== 'number' || v < 0)) {
      throw new TypeError('Invalid age');
    }
    t[p] = v;
    return true;
  }
});
proxy2.age = 30;
// proxy2.age = -5; // throws

// Example 3: Function proxy
function greet(name){ return `Hi, ${name}`;}
const proxy3 = new Proxy(greet, {
  apply(fn, thisArg, args){
    console.log('calling greet');
    return fn.apply(thisArg, args);
  }
});
console.log(proxy3('Alice')); // logs then returns

// Example 4: Revocable proxy
const { proxy: proxy4, revoke } = Proxy.revocable({}, {
  get(){ return 'revoked?'; }
});
console.log(proxy4.any); // 'revoked?'
revoke();
try { console.log(proxy4.any) } catch(e) { console.log('revoked error'); }

// Example 5: enumerate trap with ownKeys
const target5 = { a:1, b:2 };
const proxy5 = new Proxy(target5, {
  ownKeys(){ return ['b']; }
});
console.log(Object.keys(proxy5)); // ['b']


// 3. Decorators (Proposals)
// --------------------------

// Note: requires experimental support (@babel/plugin-proposal-decorators)

// Example 1: Class decorator
function sealed(constructor){
  Object.seal(constructor);
  Object.seal(constructor.prototype);
}
@sealed
class C1 {}

// Example 2: Method decorator
function log(target, key, desc){
  const orig = desc.value;
  desc.value = function(...args){
    console.log(`Call ${key}`, args);
    return orig.apply(this, args);
  };
  return desc;
}
class C2 {
  @log
  add(a,b){ return a+b; }
}
new C2().add(2,3);

// Example 3: Property decorator
function readonly(target, prop){
  Object.defineProperty(target, prop, {
    writable: false
  });
}
class C3 {
  @readonly
  name = 'immutable';
}
const c3 = new C3();
// c3.name = 'change'; // silently fails or throws in strict mode

// Example 4: Parameter decorator
function required(target, key, index){
  const meta = Reflect.getOwnMetadata('required', target, key) || [];
  meta.push(index);
  Reflect.defineMetadata('required', meta, target, key);
}
class C4 {
  greet(@required name){ return `Hello ${name}`; }
}
try {
  const meta = Reflect.getOwnMetadata('required', C4.prototype, 'greet');
  console.log(meta); // [0]
} catch(e){}

// Example 5: Decorator composition
function decoA(t,k,d){ console.log('A'); return d; }
function decoB(t,k,d){ console.log('B'); return d; }
class C5 {
  @decoA
  @decoB
  method(){}
}
// order: B then A


// 4. Dynamic Code (eval, new Function)
// -------------------------------------

// Example 1: eval for expression
const code1 = '2 + 3';
console.log(eval(code1)); // 5

// Example 2: new Function constructor
const adder = new Function('a','b','return a+b;');
console.log(adder(4,6)); // 10

// Example 3: sandboxed eval via VM module (Node.js)
const vm = require('vm');
const sandbox = { x:1 };
vm.createContext(sandbox);
vm.runInContext('x = x + 2', sandbox);
console.log(sandbox.x); // 3

// Example 4: dynamic code with template
function makeMultiplier(n){
  return new Function('x', `return x * ${n};`);
}
const times5 = makeMultiplier(5);
console.log(times5(3)); // 15

// Example 5: error handling in eval
try {
  eval('foo++;');
} catch(e) {
  console.log('Eval error:', e.message);
}


// 5. AST Manipulation & Babel Plugin Authoring
// --------------------------------------------

const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require('@babel/types');

// Example 1: Parse and inspect AST
const ast1 = parser.parse('const x = 1;');
console.log(ast1.program.body[0].declarations[0].id.name); // x

// Example 2: Transform all identifiers "a" -> "b"
const ast2 = parser.parse('function f(a){ return a + 1;}');
traverse(ast2, {
  Identifier(path){
    if(path.node.name === 'a') path.node.name = 'b';
  }
});
console.log(generate(ast2).code); // function f(b){return b+1;}

// Example 3: Insert console.log before return
const ast3 = parser.parse('function f(x){ return x*x; }');
traverse(ast3, {
  ReturnStatement(path){
    path.insertBefore(t.expressionStatement(
      t.callExpression(t.memberExpression(t.identifier('console'),'log'), [path.node.argument])
    ));
  }
});
console.log(generate(ast3).code);

// Example 4: Simple Babel plugin to prefix function names
function prefixFunctionNames() {
  return {
    visitor: {
      FunctionDeclaration(path) {
        path.node.id.name = 'prefixed_' + path.node.id.name;
      }
    }
  };
}
const ast4 = parser.parse('function test(){return;}');
traverse(ast4, prefixFunctionNames().visitor);
console.log(generate(ast4).code); // function prefixed_test(){}

// Example 5: Babel plugin that removes debugger statements
function removeDebuggerPlugin() {
  return {
    visitor: {
      DebuggerStatement(path) {
        path.remove();
      }
    }
  };
}
const ast5 = parser.parse('function f(){ debugger; return; }');
traverse(ast5, removeDebuggerPlugin().visitor);
console.log(generate(ast5).code); // function f(){return;}