"""
Todo List Application - Core Module
Provides TodoList class with CRUD operations and JSON persistence
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class TodoList:
    """Manages a list of todo items with persistence to JSON file"""
    
    def __init__(self, data_file: str = "todos.json"):
        """
        Initialize TodoList with data file path
        
        Args:
            data_file: Path to JSON file for storing todos
        """
        self.data_file = data_file
        self.todos: List[Dict] = []
        self.load()
    
    def load(self) -> None:
        """Load todos from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.todos = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load todos from {self.data_file}: {e}")
                self.todos = []
        else:
            self.todos = []
    
    def save(self) -> None:
        """Save todos to JSON file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.todos, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error: Could not save todos to {self.data_file}: {e}")
    
    def add(self, task: str, priority: str = "medium") -> Dict:
        """
        Add a new todo item
        
        Args:
            task: Description of the task
            priority: Priority level (low, medium, high)
        
        Returns:
            The created todo item
        """
        todo_id = self._generate_id()
        todo = {
            "id": todo_id,
            "task": task,
            "completed": False,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        self.todos.append(todo)
        self.save()
        return todo
    
    def list_all(self, show_completed: bool = True) -> List[Dict]:
        """
        List all todos
        
        Args:
            show_completed: Whether to include completed todos
        
        Returns:
            List of todo items
        """
        if show_completed:
            return self.todos
        return [todo for todo in self.todos if not todo["completed"]]
    
    def get_by_id(self, todo_id: int) -> Optional[Dict]:
        """
        Get a todo by ID
        
        Args:
            todo_id: ID of the todo
        
        Returns:
            Todo item or None if not found
        """
        for todo in self.todos:
            if todo["id"] == todo_id:
                return todo
        return None
    
    def complete(self, todo_id: int) -> bool:
        """
        Mark a todo as completed
        
        Args:
            todo_id: ID of the todo to complete
        
        Returns:
            True if successful, False if todo not found
        """
        todo = self.get_by_id(todo_id)
        if todo:
            todo["completed"] = True
            todo["completed_at"] = datetime.now().isoformat()
            self.save()
            return True
        return False
    
    def uncomplete(self, todo_id: int) -> bool:
        """
        Mark a todo as not completed
        
        Args:
            todo_id: ID of the todo to uncomplete
        
        Returns:
            True if successful, False if todo not found
        """
        todo = self.get_by_id(todo_id)
        if todo:
            todo["completed"] = False
            todo["completed_at"] = None
            self.save()
            return True
        return False
    
    def delete(self, todo_id: int) -> bool:
        """
        Delete a todo
        
        Args:
            todo_id: ID of the todo to delete
        
        Returns:
            True if successful, False if todo not found
        """
        for i, todo in enumerate(self.todos):
            if todo["id"] == todo_id:
                self.todos.pop(i)
                self.save()
                return True
        return False
    
    def clear_completed(self) -> int:
        """
        Remove all completed todos
        
        Returns:
            Number of todos removed
        """
        initial_count = len(self.todos)
        self.todos = [todo for todo in self.todos if not todo["completed"]]
        removed_count = initial_count - len(self.todos)
        if removed_count > 0:
            self.save()
        return removed_count
    
    def update_task(self, todo_id: int, new_task: str) -> bool:
        """
        Update the task description
        
        Args:
            todo_id: ID of the todo to update
            new_task: New task description
        
        Returns:
            True if successful, False if todo not found
        """
        todo = self.get_by_id(todo_id)
        if todo:
            todo["task"] = new_task
            self.save()
            return True
        return False
    
    def update_priority(self, todo_id: int, priority: str) -> bool:
        """
        Update the priority of a todo
        
        Args:
            todo_id: ID of the todo to update
            priority: New priority (low, medium, high)
        
        Returns:
            True if successful, False if todo not found
        """
        if priority not in ["low", "medium", "high"]:
            return False
        
        todo = self.get_by_id(todo_id)
        if todo:
            todo["priority"] = priority
            self.save()
            return True
        return False
    
    def _generate_id(self) -> int:
        """Generate a unique ID for a new todo"""
        if not self.todos:
            return 1
        return max(todo["id"] for todo in self.todos) + 1
    
    def get_stats(self) -> Dict:
        """
        Get statistics about todos
        
        Returns:
            Dictionary with stats (total, completed, pending)
        """
        total = len(self.todos)
        completed = sum(1 for todo in self.todos if todo["completed"])
        pending = total - completed
        
        return {
            "total": total,
            "completed": completed,
            "pending": pending
        }
