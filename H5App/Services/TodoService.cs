using H5App.Models;
namespace H5App.Services;

public class TodoService : ITodoService
{
    private List<TodoItem> _items;
    private int _nextId = 1;
    public TodoService(){
        _items = new List<TodoItem> {
            new TodoItem { Id = _nextId++, Description = "Get good", CreatedTime = DateTime.Now.AddDays(-1), Priority = PriorityLevel.Normal, Completed = false },
            new TodoItem { Id = _nextId++, Description = "Do homework", CreatedTime = DateTime.Now.AddDays(-2), Priority = PriorityLevel.High, Completed = false },
            new TodoItem { Id = _nextId++, Description = "Find the answer to life", CreatedTime = DateTime.Now.AddDays(-3), Priority = PriorityLevel.Low, Completed = true }
        };
    }

    public async Task<TodoItem> AddTodoItemAsync(TodoItem item)
    {
        await Task.Delay(100);
            item.Id = _nextId++;
            item.CreatedTime = DateTime.Now;
            _items.Add(item);
            return item;
    }

    public async Task<bool> DeleteTodoItemAsync(int id)
    {
        var item = _items.FirstOrDefault(i => i.Id == id);
        if(item != null) {
            _items.Remove(item);
            return true;
        }
        return false;
    }

    public async Task<List<TodoItem>> GetAllTodoItemsAsync()
    {
        return _items.ToList();
    }

    public async Task<TodoItem> GetTodoItemByIdAsync(int id)
    {
        return _items.FirstOrDefault(i => i.Id == id);
    }

    public async Task<TodoItem> UpdateTodoItemAsync(TodoItem item)
    {
        var existingItem = _items.FirstOrDefault(i => i.Id == item.Id);
        if(existingItem != null){
            existingItem.Description = item.Description;
            existingItem.Priority = item.Priority;
            existingItem.Completed = item.Completed;
        }
        return existingItem;
    }
}