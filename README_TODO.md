# Todo List Application

A simple, efficient command-line todo list manager built with Python.

## Features

- ✅ Add, list, complete, and delete todos
- 📊 Priority levels (low, medium, high)
- 💾 Persistent storage with JSON
- 📈 Statistics and progress tracking
- 🎨 Clean CLI interface with visual indicators

## Installation

No additional dependencies required beyond Python 3.6+. The application uses only standard library modules.

## Usage

### Basic Commands

**Add a new todo:**
```bash
python main.py add "Buy groceries"
python main.py add "Finish project report" -p high
```

**List all todos:**
```bash
python main.py list
```

**List only pending todos:**
```bash
python main.py list -p
```

**Complete a todo:**
```bash
python main.py complete 1
```

**Mark a todo as pending again:**
```bash
python main.py uncomplete 1
```

**Delete a todo:**
```bash
python main.py delete 1
```

**Update a todo:**
```bash
python main.py update 1 -t "New task description"
python main.py update 1 -p high
```

**Clear all completed todos:**
```bash
python main.py clear
```

**Show statistics:**
```bash
python main.py stats
```

### Priority Levels

- `low` (↓): Low priority tasks
- `medium` (→): Medium priority tasks (default)
- `high` (↑): High priority tasks

### Examples

```bash
# Add a high priority task
python main.py add "Submit tax returns" -p high

# Add multiple tasks
python main.py add "Call dentist"
python main.py add "Review pull requests" -p medium
python main.py add "Water plants" -p low

# List all todos
python main.py list

# Complete a task
python main.py complete 1

# Update task description
python main.py update 2 -t "Call dentist to reschedule appointment"

# Update priority
python main.py update 3 -p high

# View statistics
python main.py stats

# Clear completed tasks
python main.py clear
```

## Data Storage

Todos are stored in `todos.json` in the current directory. The file is automatically created on first use and updated with each operation.

## File Structure

- `todo_app.py` - Core TodoList class with CRUD operations
- `cli.py` - Command-line interface handler
- `main.py` - Application entry point
- `todos.json` - Data storage (auto-generated)

## Help

For help with any command:
```bash
python main.py -h
python main.py add -h
python main.py list -h
```

## License

This todo list application is provided as-is for personal and educational use.
