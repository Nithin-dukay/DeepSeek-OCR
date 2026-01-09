"""
Todo List Application - CLI Interface
Provides command-line interface for interacting with the todo list
"""

import argparse
import sys
from todo_app import TodoList


class TodoCLI:
    """Command-line interface for TodoList application"""
    
    def __init__(self):
        """Initialize CLI with TodoList instance"""
        self.todo_list = TodoList()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands"""
        parser = argparse.ArgumentParser(
            description="Todo List Application - Manage your tasks efficiently",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Add command
        add_parser = subparsers.add_parser('add', help='Add a new todo')
        add_parser.add_argument('task', nargs='+', help='Task description')
        add_parser.add_argument('-p', '--priority', choices=['low', 'medium', 'high'],
                               default='medium', help='Priority level (default: medium)')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List all todos')
        list_parser.add_argument('-a', '--all', action='store_true',
                                help='Show completed todos (default: show all)')
        list_parser.add_argument('-p', '--pending', action='store_true',
                                help='Show only pending todos')
        
        # Complete command
        complete_parser = subparsers.add_parser('complete', help='Mark todo as completed')
        complete_parser.add_argument('id', type=int, help='Todo ID')
        
        # Uncomplete command
        uncomplete_parser = subparsers.add_parser('uncomplete', help='Mark todo as not completed')
        uncomplete_parser.add_argument('id', type=int, help='Todo ID')
        
        # Delete command
        delete_parser = subparsers.add_parser('delete', help='Delete a todo')
        delete_parser.add_argument('id', type=int, help='Todo ID')
        
        # Update command
        update_parser = subparsers.add_parser('update', help='Update a todo')
        update_parser.add_argument('id', type=int, help='Todo ID')
        update_parser.add_argument('-t', '--task', nargs='+', help='New task description')
        update_parser.add_argument('-p', '--priority', choices=['low', 'medium', 'high'],
                                   help='New priority level')
        
        # Clear command
        subparsers.add_parser('clear', help='Clear all completed todos')
        
        # Stats command
        subparsers.add_parser('stats', help='Show statistics')
        
        return parser
    
    def run(self, args=None):
        """
        Run the CLI with provided arguments
        
        Args:
            args: Command-line arguments (uses sys.argv if None)
        """
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        # Route to appropriate handler
        command_handlers = {
            'add': self._handle_add,
            'list': self._handle_list,
            'complete': self._handle_complete,
            'uncomplete': self._handle_uncomplete,
            'delete': self._handle_delete,
            'update': self._handle_update,
            'clear': self._handle_clear,
            'stats': self._handle_stats
        }
        
        handler = command_handlers.get(parsed_args.command)
        if handler:
            handler(parsed_args)
    
    def _handle_add(self, args):
        """Handle add command"""
        task = ' '.join(args.task)
        todo = self.todo_list.add(task, args.priority)
        print(f"✓ Added todo #{todo['id']}: {task}")
        if args.priority != 'medium':
            print(f"  Priority: {args.priority}")
    
    def _handle_list(self, args):
        """Handle list command"""
        show_completed = not args.pending
        todos = self.todo_list.list_all(show_completed)
        
        if not todos:
            print("No todos found.")
            return
        
        print(f"\n{'ID':<5} {'Status':<12} {'Priority':<10} {'Task'}")
        print("-" * 70)
        
        for todo in todos:
            status = "✓ Done" if todo['completed'] else "○ Pending"
            priority_symbols = {"low": "↓", "medium": "→", "high": "↑"}
            priority_display = f"{priority_symbols.get(todo['priority'], '→')} {todo['priority']}"
            
            print(f"{todo['id']:<5} {status:<12} {priority_display:<10} {todo['task']}")
        
        print()
    
    def _handle_complete(self, args):
        """Handle complete command"""
        if self.todo_list.complete(args.id):
            todo = self.todo_list.get_by_id(args.id)
            print(f"✓ Completed todo #{args.id}: {todo['task']}")
        else:
            print(f"✗ Todo #{args.id} not found")
            sys.exit(1)
    
    def _handle_uncomplete(self, args):
        """Handle uncomplete command"""
        if self.todo_list.uncomplete(args.id):
            todo = self.todo_list.get_by_id(args.id)
            print(f"○ Marked todo #{args.id} as pending: {todo['task']}")
        else:
            print(f"✗ Todo #{args.id} not found")
            sys.exit(1)
    
    def _handle_delete(self, args):
        """Handle delete command"""
        todo = self.todo_list.get_by_id(args.id)
        if todo:
            task = todo['task']
            if self.todo_list.delete(args.id):
                print(f"✓ Deleted todo #{args.id}: {task}")
        else:
            print(f"✗ Todo #{args.id} not found")
            sys.exit(1)
    
    def _handle_update(self, args):
        """Handle update command"""
        todo = self.todo_list.get_by_id(args.id)
        if not todo:
            print(f"✗ Todo #{args.id} not found")
            sys.exit(1)
        
        updated = False
        
        if args.task:
            new_task = ' '.join(args.task)
            if self.todo_list.update_task(args.id, new_task):
                print(f"✓ Updated task for todo #{args.id}: {new_task}")
                updated = True
        
        if args.priority:
            if self.todo_list.update_priority(args.id, args.priority):
                print(f"✓ Updated priority for todo #{args.id}: {args.priority}")
                updated = True
        
        if not updated:
            print("No updates specified. Use -t for task or -p for priority.")
    
    def _handle_clear(self, args):
        """Handle clear command"""
        count = self.todo_list.clear_completed()
        if count > 0:
            print(f"✓ Cleared {count} completed todo(s)")
        else:
            print("No completed todos to clear")
    
    def _handle_stats(self, args):
        """Handle stats command"""
        stats = self.todo_list.get_stats()
        
        print("\n📊 Todo Statistics")
        print("-" * 30)
        print(f"Total todos:     {stats['total']}")
        print(f"Completed:       {stats['completed']}")
        print(f"Pending:         {stats['pending']}")
        
        if stats['total'] > 0:
            completion_rate = (stats['completed'] / stats['total']) * 100
            print(f"Completion rate: {completion_rate:.1f}%")
        
        print()


def main():
    """Main entry point for CLI"""
    cli = TodoCLI()
    cli.run()


if __name__ == '__main__':
    main()
