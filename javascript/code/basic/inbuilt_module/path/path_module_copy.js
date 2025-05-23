/**
 * Node.js 'path' Module - Comprehensive Examples
 * 
 * The 'path' module provides utilities for working with file and directory paths.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const path = require('path');

// 1. path.basename(): Get the last portion of a path
console.log('1. Basename:', path.basename('/foo/bar/baz.txt')); 
// Output: 'baz.txt'
console.log('1. Basename (with ext):', path.basename('/foo/bar/baz.txt', '.txt')); 
// Output: 'baz'

// 2. path.dirname(): Get the directory name of a path
console.log('2. Dirname:', path.dirname('/foo/bar/baz.txt')); 
// Output: '/foo/bar'

// 3. path.extname(): Get the extension of the path
console.log('3. Extname:', path.extname('/foo/bar/baz.txt')); 
// Output: '.txt'
console.log('3. Extname (no ext):', path.extname('/foo/bar/baz')); 
// Output: ''

// 4. path.join(): Join all arguments together and normalize the resulting path
console.log('4. Join:', path.join('/foo', 'bar', 'baz/asdf', 'quux', '..')); 
// Output: '/foo/bar/baz/asdf'

// 5. path.resolve(): Resolve a sequence of paths or path segments into an absolute path
console.log('5. Resolve:', path.resolve('foo/bar', '/tmp/file/', '..', 'a/../subfile')); 
// Output: '/tmp/subfile' (on Unix-like systems)

// 6. path.normalize(): Normalize a string path, resolving '..' and '.' segments
console.log('6. Normalize:', path.normalize('/foo/bar//baz/asdf/quux/..')); 
// Output: '/foo/bar/baz/asdf'

// 7. path.isAbsolute(): Test if a path is absolute
console.log('7. Is Absolute (/foo/bar):', path.isAbsolute('/foo/bar')); 
// Output: true
console.log('7. Is Absolute (foo/bar):', path.isAbsolute('foo/bar')); 
// Output: false

// 8. path.relative(): Get the relative path from one path to another
console.log('8. Relative:', path.relative('/data/orandea/test/aaa', '/data/orandea/impl/bbb')); 
// Output: '../../impl/bbb'

// 9. path.parse(): Parse a path into root, dir, base, ext, and name
const parsed = path.parse('/home/user/dir/file.txt');
console.log('9. Parse:', parsed); 
// Output: { root: '/', dir: '/home/user/dir', base: 'file.txt', ext: '.txt', name: 'file' }

// 10. path.format(): Format a path object into a path string
const formatted = path.format({
    dir: '/home/user/dir',
    base: 'file.txt'
});
console.log('10. Format:', formatted); 
// Output: '/home/user/dir/file.txt'

// 11. path.sep: Platform-specific path segment separator
console.log('11. Path separator:', JSON.stringify(path.sep)); 
// Output: '/' (POSIX) or '\\' (Windows)

// 12. path.delimiter: Platform-specific path delimiter
console.log('12. Path delimiter:', JSON.stringify(path.delimiter)); 
// Output: ':' (POSIX) or ';' (Windows)

// 13. path.posix & path.win32: Explicit POSIX/Windows path methods
console.log('13. POSIX join:', path.posix.join('/foo', 'bar', 'baz')); 
// Output: '/foo/bar/baz'
console.log('13. WIN32 join:', path.win32.join('C:\\foo', 'bar', 'baz')); 
// Output: 'C:\\foo\\bar\\baz'

// 14. Exception Handling: Invalid path input
try {
    path.join(null, 'foo');
} catch (err) {
    console.log('14. Exception caught:', err.message); 
    // Output: Path must be a string. Received null
}

// 15. Edge Case: path.basename with trailing slash
console.log('15. Basename (trailing slash):', path.basename('/foo/bar/')); 
// Output: 'bar'

/**
 * Additional Notes:
 * - All major and minor methods of 'path' module are covered.
 * - Both POSIX and Windows-specific usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js path module.
 */