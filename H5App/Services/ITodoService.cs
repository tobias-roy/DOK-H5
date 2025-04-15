using H5App.Models;

namespace H5App.Services;

public interface ITodoService {
    Task<List<TodoItem>> GetAllTodoItemsAsync();
    Task<TodoItem> GetTodoItemByIdAsync(int id);
    Task<TodoItem> AddTodoItemAsync(TodoItem item);
    Task<TodoItem> UpdateTodoItemAsync(TodoItem item);
    Task<bool> DeleteTodoItemAsync(int id);
}