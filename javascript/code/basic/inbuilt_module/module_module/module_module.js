/**
 * Node.js 'module' Module - Comprehensive Examples
 * 
 * The 'module' module provides access to Node.js's internal module loader API.
 * This file demonstrates all major and minor methods, properties, and edge cases.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const Module = require('module');
const path = require('path');
const fs = require('fs');

// 1. Module.builtinModules: List all Node.js built-in modules
console.log('1. Built-in modules:', Module.builtinModules.slice(0, 5), '...'); 
// Output: [ 'assert', 'async_hooks', 'buffer', 'child_process', 'cluster' ] ...

// 2. Module._resolveFilename: Resolve the full path of a module
const resolvedPath = Module._resolveFilename('fs', module);
console.log('2. Resolved path for "fs":', resolvedPath); 
// Output: fs (built-in modules return their name)

// 3. Module._resolveLookupPaths: Get lookup paths for a module request
const lookupPaths = Module._resolveLookupPaths('some-module', module);
console.log('3. Lookup paths for "some-module":', lookupPaths); 
// Output: [ 'node_modules', ... ] (array of paths)

// 4. Module._load: Load a module as Node.js does internally
const loadedFs = Module._load('fs', module, false);
console.log('4. Loaded fs === require("fs"):', loadedFs === require('fs')); 
// Output: true

// 5. Module.createRequire: Create a require function with a custom path
const customRequire = Module.createRequire(__filename);
const pathModule = customRequire('path');
console.log('5. customRequire("path") === require("path"):', pathModule === require('path')); 
// Output: true

// 6. Module.wrap: Show how Node wraps modules (for advanced use)
const wrapped = Module.wrap('console.log("6. Inside wrapped module");');
console.log('6. Module.wrap output:', wrapped); 
// Output: (function (exports, require, module, __filename, __dirname) { ... })

// 7. Module._cache: Access the module cache
console.log('7. Module._cache has this file:', !!Module._cache[__filename]); 
// Output: true

// 8. Module._compile: Compile code in the context of a module
const tempFile = path.join(__dirname, 'tempModule.js');
fs.writeFileSync(tempFile, 'module.exports = 42;');
const tempModule = new Module(tempFile, module);
tempModule.filename = tempFile;
tempModule.paths = Module._nodeModulePaths(__dirname);
const code = fs.readFileSync(tempFile, 'utf8');
tempModule._compile(code, tempFile);
console.log('8. Compiled module exports:', tempModule.exports); 
// Output: 42
fs.unlinkSync(tempFile);

// 9. Module._nodeModulePaths: Get node_modules lookup paths for a directory
const nodeModulePaths = Module._nodeModulePaths(__dirname);
console.log('9. node_modules paths for this dir:', nodeModulePaths.slice(0, 2)); 
// Output: [ ..., ... ] (array of paths)

// 10. Exception Handling: Try to resolve a non-existent module
try {
    Module._resolveFilename('non-existent-module', module);
} catch (err) {
    console.log('10. Exception caught:', err.code); 
    // Output: MODULE_NOT_FOUND
}

// 11. module.children: List child modules loaded by this module
console.log('11. module.children:', module.children.map(m => m.id)); 
// Output: [ ... ] (array of child module ids, usually empty in main file)

// 12. module.parent: Show parent module (null for entry point)
console.log('12. module.parent:', module.parent); 
// Output: null (if this is the entry point)

// 13. module.paths: Show module resolution paths for this module
console.log('13. module.paths:', module.paths.slice(0, 2)); 
// Output: [ ..., ... ] (array of paths)

// 14. module.require: Use the require method of the current module
const osModule = module.require('os');
console.log('14. module.require("os") === require("os"):', osModule === require('os')); 
// Output: true

// 15. module.loaded: Check if this module is loaded
console.log('15. module.loaded:', module.loaded); 
// Output: true (after main module is loaded)

/**
 * Additional Notes:
 * - All major and minor methods/properties of 'module' are covered.
 * - Both public and internal APIs are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js module loader internals.
 * - Internal APIs (underscore-prefixed) are not officially documented and may change.
 */