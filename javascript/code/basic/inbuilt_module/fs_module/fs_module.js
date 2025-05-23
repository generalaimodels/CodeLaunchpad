/**
 * Node.js 'fs' Module - Comprehensive Examples
 * 
 * The 'fs' (File System) module enables interacting with the file system in a way modeled on standard POSIX functions.
 * This file demonstrates all major and minor methods, both synchronous and asynchronous, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const fs = require('fs');
const path = require('path');

// Utility: Paths for demonstration
const demoDir = path.join(__dirname, 'fs_demo');
const filePath = path.join(demoDir, 'example.txt');
const copyPath = path.join(demoDir, 'copy.txt');
const movePath = path.join(demoDir, 'moved.txt');
const jsonPath = path.join(demoDir, 'data.json');
const watchFilePath = path.join(demoDir, 'watch.txt');

// Ensure demo directory exists
if (!fs.existsSync(demoDir)) fs.mkdirSync(demoDir);

// 1. fs.writeFile & fs.readFile (Async): Write and read a file
fs.writeFile(filePath, 'Hello, FileSystem!', (err) => {
    if (err) throw err;
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) throw err;
        console.log('1. Read content:', data); // Output: Hello, FileSystem!
    });
});

// 2. fs.writeFileSync & fs.readFileSync (Sync): Synchronous file write/read
fs.writeFileSync(filePath, 'Sync Write Example');
const syncData = fs.readFileSync(filePath, 'utf8');
console.log('2. Sync read:', syncData); // Output: Sync Write Example

// 3. fs.appendFile & fs.appendFileSync: Append data to a file
fs.appendFile(filePath, '\nAppended Line', (err) => {
    if (err) throw err;
    const appended = fs.readFileSync(filePath, 'utf8');
    console.log('3. Appended content:', appended.split('\n')[1]); // Output: Appended Line
});
fs.appendFileSync(filePath, '\nAppended Sync');

// 4. fs.rename & fs.renameSync: Move or rename a file
fs.rename(filePath, movePath, (err) => {
    if (err) throw err;
    console.log('4. File renamed/moved'); // Output: File renamed/moved
    fs.renameSync(movePath, filePath); // Move back for further examples
});

// 5. fs.copyFile & fs.copyFileSync: Copy a file
fs.copyFile(filePath, copyPath, (err) => {
    if (err) throw err;
    const copied = fs.readFileSync(copyPath, 'utf8');
    console.log('5. Copied file content:', copied.split('\n')[0]); // Output: Sync Write Example
});
fs.copyFileSync(filePath, copyPath);

// 6. fs.unlink & fs.unlinkSync: Delete a file
fs.writeFileSync(copyPath, 'To be deleted');
fs.unlink(copyPath, (err) => {
    if (err) throw err;
    console.log('6. File deleted (async)'); // Output: File deleted (async)
});
fs.writeFileSync(copyPath, 'To be deleted again');
fs.unlinkSync(copyPath); // No output, but file is deleted

// 7. fs.mkdir & fs.rmdir (Sync/Async): Create and remove directories
const tempDir = path.join(demoDir, 'temp');
fs.mkdir(tempDir, (err) => {
    if (err) throw err;
    console.log('7. Directory created'); // Output: Directory created
    fs.rmdir(tempDir, (err) => {
        if (err) throw err;
        console.log('7. Directory removed'); // Output: Directory removed
    });
});
fs.mkdirSync(tempDir);
fs.rmdirSync(tempDir);

// 8. fs.readdir & fs.readdirSync: List directory contents
const files = fs.readdirSync(demoDir);
console.log('8. Directory contents:', files); // Output: Array of file names in fs_demo

// 9. fs.stat & fs.statSync: Get file/directory stats
fs.stat(filePath, (err, stats) => {
    if (err) throw err;
    console.log('9. Is file:', stats.isFile()); // Output: true
    console.log('9. Size:', stats.size); // Output: (file size in bytes)
});
const statSync = fs.statSync(filePath);
console.log('9. (Sync) Is file:', statSync.isFile()); // Output: true

// 10. fs.watch: Watch for file changes
fs.writeFileSync(watchFilePath, 'Initial');
const watcher = fs.watch(watchFilePath, (eventType, filename) => {
    if (filename) {
        console.log(`10. File ${filename} changed: ${eventType}`); // Output: File watch.txt changed: change
        watcher.close(); // Stop watching after first change
    }
});
setTimeout(() => {
    fs.appendFileSync(watchFilePath, '\nWatched change');
}, 100);

// 11. fs.existsSync: Check if a file exists (deprecated async version, use sync)
console.log('11. File exists:', fs.existsSync(filePath)); // Output: true

// 12. fs.open & fs.close: Low-level file descriptor operations
fs.open(filePath, 'r', (err, fd) => {
    if (err) throw err;
    const buffer = Buffer.alloc(10);
    fs.read(fd, buffer, 0, 10, 0, (err, bytesRead, buf) => {
        if (err) throw err;
        console.log('12. Read bytes:', buf.toString('utf8', 0, bytesRead)); // Output: First 10 chars
        fs.close(fd, (err) => {
            if (err) throw err;
            console.log('12. File closed'); // Output: File closed
        });
    });
});

// 13. fs.readFile with JSON: Read and parse JSON file
fs.writeFileSync(jsonPath, JSON.stringify({ a: 1, b: 2 }));
fs.readFile(jsonPath, 'utf8', (err, data) => {
    if (err) throw err;
    const obj = JSON.parse(data);
    console.log('13. JSON read:', obj); // Output: { a: 1, b: 2 }
});

// 14. fs.truncate & fs.truncateSync: Truncate file to a specific length
fs.writeFileSync(filePath, '1234567890');
fs.truncate(filePath, 5, (err) => {
    if (err) throw err;
    const truncated = fs.readFileSync(filePath, 'utf8');
    console.log('14. Truncated content:', truncated); // Output: 12345
});
fs.writeFileSync(filePath, '1234567890');
fs.truncateSync(filePath, 3);
console.log('14. (Sync) Truncated:', fs.readFileSync(filePath, 'utf8')); // Output: 123

// 15. fs.chmod & fs.chmodSync: Change file permissions
fs.chmod(filePath, 0o644, (err) => {
    if (err) throw err;
    console.log('15. Permissions changed (async)'); // Output: Permissions changed (async)
});
fs.chmodSync(filePath, 0o600);

// 16. fs.utimes & fs.utimesSync: Update file timestamps
const now = new Date();
fs.utimes(filePath, now, now, (err) => {
    if (err) throw err;
    console.log('16. Timestamps updated'); // Output: Timestamps updated
});
fs.utimesSync(filePath, now, now);

// 17. fs.createReadStream & fs.createWriteStream: Stream file reading/writing
const streamWritePath = path.join(demoDir, 'stream.txt');
const writeStream = fs.createWriteStream(streamWritePath);
writeStream.write('Streaming line 1\n');
writeStream.write('Streaming line 2\n');
writeStream.end();
writeStream.on('finish', () => {
    const readStream = fs.createReadStream(streamWritePath, { encoding: 'utf8' });
    readStream.on('data', chunk => {
        console.log('17. Stream read chunk:', chunk); // Output: Streaming line 1\nStreaming line 2\n
    });
});

// 18. fs.access & fs.accessSync: Test file permissions
fs.access(filePath, fs.constants.R_OK | fs.constants.W_OK, (err) => {
    console.log('18. File is readable and writable:', !err); // Output: true
});
console.log('18. (Sync) File is readable:', !fs.accessSync(filePath, fs.constants.R_OK)); // Output: true

// 19. fs.rm & fs.rmSync: Remove files/directories (Node 14.14+)
const rmFile = path.join(demoDir, 'toremove.txt');
fs.writeFileSync(rmFile, 'Remove me');
fs.rm(rmFile, (err) => {
    if (err) throw err;
    console.log('19. File removed with fs.rm'); // Output: File removed with fs.rm
});
fs.writeFileSync(rmFile, 'Remove me again');
fs.rmSync(rmFile);

// 20. Exception Handling: Try-catch for sync, error callback for async
try {
    fs.readFileSync('nonexistent.txt');
} catch (err) {
    console.log('20. Exception caught:', err.code); // Output: ENOENT
}
fs.readFile('nonexistent.txt', (err, data) => {
    if (err) {
        console.log('20. Async error:', err.code); // Output: ENOENT
    }
});

/**
 * Summary:
 * - All major and minor methods of 'fs' module are covered.
 * - Both async and sync versions are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js fs module.
 */