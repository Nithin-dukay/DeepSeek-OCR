#!/usr/bin/env python3
"""
Simple CLI Todo List Application
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any


class TodoApp:
    def __init__(self, data_file: str = "tasks.json"):
        self.data_file = data_file
        self.tasks = self.load_tasks()

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Could not load tasks file. Starting with empty list.")
                return []
        return []

    def save_tasks(self) -> None:
        """Save tasks to JSON file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.tasks, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def add_task(self, description: str) -> None:
        """Add a new task"""
        task_id = max([task.get('id', 0) for task in self.tasks], default=0) + 1
        task = {
            'id': task_id,
            'description': description,
            'completed': False,
            'created': datetime.now()
        }
        self.tasks.append(task)
        self.save_tasks()
        print(f"Task added: {description}")

    def list_tasks(self, show_completed: bool = False) -> None:
        """List all tasks"""
        if not self.tasks:
            print("No tasks found.")
            return

        for task in self.tasks:
            if not show_completed and task['completed']:
                continue
            status = "[✓]" if task['completed'] else "[ ]"
            print(f"{task['id']}. {status} {task['description']}")

    def complete_task(self, task_id: int) -> None:
        """Mark a task as completed"""
        for task in self.tasks:
            if task['id'] == task_id:
                if task['completed']:
                    print(f"Task {task_id} is already completed.")
                else:
                    task['completed'] = True
                    self.save_tasks()
                    print(f"Task {task_id} marked as completed.")
                return
        print(f"Task {task_id} not found.")

    def delete_task(self, task_id: int) -> None:
        """Delete a task"""
        for i, task in enumerate(self.tasks):
            if task['id'] == task_id:
                removed_task = self.tasks.pop(i)
                self.save_tasks()
                print(f"Task deleted: {removed_task['description']}")
                return
        print(f"Task {task_id} not found.")


def main():
    parser = argparse.ArgumentParser(description="Simple CLI Todo List App")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add task
    add_parser = subparsers.add_parser('add', help='Add a new task')
    add_parser.add_argument('description', help='Task description')

    # List tasks
    list_parser = subparsers.add_parser('list', help='List tasks')
    list_parser.add_argument('--all', action='store_true', help='Show completed tasks too')

    # Complete task
    complete_parser = subparsers.add_parser('complete', help='Mark task as completed')
    complete_parser.add_argument('task_id', type=int, help='Task ID to complete')

    # Delete task
    delete_parser = subparsers.add_parser('delete', help='Delete a task')
    delete_parser.add_argument('task_id', type=int, help='Task ID to delete')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    app = TodoApp()

    if args.command == 'add':
        app.add_task(args.description)
    elif args.command == 'list':
        app.list_tasks(show_completed=args.all)
    elif args.command == 'complete':
        app.complete_task(args.task_id)
    elif args.command == 'delete':
        app.delete_task(args.task_id)


if __name__ == "__main__":
    main()