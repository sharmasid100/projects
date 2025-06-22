
# 🦥 Sloth Language

**Sloth** is a custom interpreted programming language built from scratch in Python.  
It features a lexer, parser, semantic analyzer, and interpreter, all designed for `.slw` source files.  
It also includes an interactive REPL for quick experimentation.

---

## 📦 Features

- 🧾 Custom syntax with `.slw` file extension
- ⚙️ Lexer and parser using [PLY (Python Lex-Yacc)](https://www.dabeaz.com/ply/)
- 🧠 Semantic analyzer with scoped symbol tables
- 🏃 Interpreter to execute the AST
- 🔁 Support for:
  - Variables and expressions
  - Functions and control flow (if , repeat)
  - Arrays, booleans, strings, numbers
- 💬 REPL mode for interactive execution
- 📂 CLI interface with `sloth` command

---

## 🛠️ Installation

Clone and install locally using `setuptools`:

```bash
git clone https://github.com/sharmasid100/slothlang
cd slothlang
pip install .
````

This installs a CLI tool named `sloth`.

---

## 🚀 Usage

### ▶️ Run a Sloth file

sloth path/to/file.slw

### 💬 Start the REPL


sloth
# or
sloth --repl

---

## 📝 Example Code (`test.slw`)

```sloth
<< This is a comment >>

fn greet(string name) {
    print("Hello, " + name)
}

greet("Sloth")
```

---

## 📁 Project Structure

```
sloth/
├── lexer.py                # Tokenizer
├── parser.py               # Grammar definition
├── parsetab.py             # PLY-generated tables (keep this)
├── interpreter.py          # Executes AST
├── semantic_analyzer.py    # Variable and type checking
├── run.py                  # Runs .slw files
├── repl.py                 # REPL loop
└── __main__.py             # CLI entry point (sloth command)
```

---

## ⚠️ Notes

* `parsetab.py` is required and included — don’t delete it.
* `parser.out` is not needed and should be ignored (used for debugging only).
* Add this to `.gitignore` if needed:

  ```
  parser.out
  ```

---

## 📜 License

MIT License — use freely, build wildly 🦥
