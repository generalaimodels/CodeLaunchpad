
| Symbol                       | Use / Meaning                                                      |                                             |
| ---------------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| `+`                          | Addition (arithmetic), also string/list concatenation              |                                             |
| `-`                          | Subtraction (arithmetic), unary negation                           |                                             |
| `*`                          | Multiplication (arithmetic), unpacking operator (`*args`)          |                                             |
| `/`                          | True division (returns float)                                      |                                             |
| `//`                         | Floor division (truncates toward negative infinity)                |                                             |
| `%`                          | Modulus (remainder)                                                |                                             |
| `**`                         | Exponentiation                                                     |                                             |
| `=`                          | Assignment of value                                                |                                             |
| `+=`, `-=`, `*=`, `/=`, etc. | Augmented assignment (e.g. `x += 1` is `x = x + 1`)                |                                             |
| `==`                         | Equality comparison                                                |                                             |
| `!=`                         | Inequality comparison                                              |                                             |
| `>`, `<`, `>=`, `<=`         | Greater/less than (and or equal) comparisons                       |                                             |
| `and`, `or`, `not`           | Logical operators                                                  |                                             |
| `is`, `is not`               | Identity comparison (same object in memory)                        |                                             |
| `in`, `not in`               | Membership test (e.g. `x in list`)                                 |                                             |
| `&`, \`                      | `, `^`, `\~`, `<<`, `>>\`                                          | Bitwise AND, OR, XOR, NOT, left/right shift |
| `:`                          | Used in slicing (`a[1:4]`), to start indented blocks (`def f():`)  |                                             |
| `,`                          | Separator in lists, tuples, function arguments                     |                                             |
| `;`                          | Statement separator (rarely used)                                  |                                             |
| `.`                          | Attribute or method access (`obj.attr`)                            |                                             |
| `()`                         | Function call, grouping of expressions                             |                                             |
| `[]`                         | List literals, indexing, slicing                                   |                                             |
| `{}`                         | Dict/set literals                                                  |                                             |
| `#`                          | Start of a comment                                                 |                                             |
| `''' … '''`, `""" … """`     | Multiline string literals / docstrings                             |                                             |
| `\`                          | Line continuation                                                  |                                             |
| `@`                          | Decorator marker (e.g. `@staticmethod`)                            |                                             |
| `->`                         | Function return-type annotation (e.g. `def f() -> int:`)           |                                             |
| `:=`                         | “Walrus” operator: assignment expression (e.g. `if (n := f()):`)   |                                             |
| `...`                        | Ellipsis literal (used in slicing, e.g. multi-dim arrays)          |                                             |
| `_`                          | Conventional “throwaway” variable; also holds the last REPL result |                                             |

---


| Escape Sequence   | Meaning / Effect                                            |
| ----------------- | ----------------------------------------------------------- |
| `\n`              | Line feed (newline)                                         |
| `\r`              | Carriage return                                             |
| `\t`              | Horizontal tab                                              |
| `\b`              | Backspace                                                   |
| `\f`              | Form feed                                                   |
| `\v`              | Vertical tab                                                |
| `\a`              | Bell/alert (may make a sound)                               |
| `\0`              | Null character                                              |
| `\\`              | Literal backslash (`\`)                                     |
| `\'`              | Single quote (use inside single-quoted strings)             |
| `\"`              | Double quote (use inside double-quoted strings)             |
| `\ooo`            | Character with octal value `ooo` (1–3 octal digits)         |
| `\xhh`            | Character with hex value `hh` (exactly 2 hex digits)        |
| `\uXXXX`          | Unicode character with 16-bit hex value `XXXX`              |
| `\UXXXXXXXX`      | Unicode character with 32-bit hex value `XXXXXXXX`          |
| `\N{name}`        | Unicode character by its official name (e.g. `\N{SNOWMAN}`) |
| `\` (at line end) | Line-continuation (ignores the newline that follows)        |

**Notes:**

* In raw strings (prefixed with `r` or `R`), most escapes are suppressed; e.g. `r"\n"` represents a literal backslash-n, not a newline.
* Octal/hex escapes must use exactly the indicated number of digits (e.g. `\x0A`, not `\xA`).
* Unicode escapes let you embed virtually any character by code point.


The “walrus” operator `:=` was introduced in Python 3.8 to let you assign to a variable as part of an expression. In essence, it combines an assignment and a value-returning expression in one. Here’s a simple illustrative snippet:

```python
# Without walrus
data = input("Enter a number (or blank to stop): ")
while data != "":
    num = int(data)
    print(f"Twice {num} is {2*num}")
    data = input("Enter a number (or blank to stop): ")
```

With the walrus operator you can write it more concisely:

```python
# With walrus
while (data := input("Enter a number (or blank to stop): ")) != "":
    num = int(data)
    print(f"Twice {num} is {2*num}")
```

### What’s happening here?

1. **Expression and assignment in one**

   * `(data := input(...))` calls `input()`, assigns its result to `data`, *and* yields that same result for the surrounding `while` test.
2. **Loop condition**

   * The `while ... != ""` then checks if `data` is non-empty.
3. **Body**

   * Inside the loop, `data` is already set, so you can immediately convert it and use it.

---

#### Another example: filtering in a comprehension

Suppose you want to collect only the even squared values from a list:

```python
nums = [1, 2, 3, 4, 5, 6]
evens = [square for n in nums if (square := n*n) % 2 == 0]
print(evens)   # [4, 16, 36]
```

* Here, `(square := n*n)` computes `n*n`, binds it to `square`, *and* then `% 2 == 0` tests whether that result is even.
* Without the walrus you’d have to compute `n*n` twice or introduce a separate loop.

---

### When to use `:=`

* **Avoid redundant work**: when you’d otherwise compute the same expression twice.
* **Tighten loops**: read-test loops (e.g. reading lines, streams, user input).
* **Keep code compact**: but beware of hurting readability—overusing it can make expressions dense.

**Tip:** If you find a complex expression inside your `if`/`while` or comprehension that you need both to test and to use, the walrus can cleanly assign it in-line.
