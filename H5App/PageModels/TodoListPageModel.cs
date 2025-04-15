using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using H5App.Messages;
using H5App.Models;
using H5App.PageModels;
using H5App.Services;

namespace H5App.PageModels;

public partial class TodoListPageModel : BasePageModel {
    private readonly ITodoService _todoService;

    public ObservableCollection<TodoItem> TodoItems {get; set;}

    [ObservableProperty]
    private partial bool _isRefreshing {get; set;}

    public TodoListPageModel(ITodoService todoService){
        _todoService = todoService;
        TodoItems = new ObservableCollection<TodoItem>();

        WeakReferenceMessenger.Default.Register<DisplayMessage>(this, async (r, m) => {
            await Shell.Current.DisplayAlert("You just deleted this task:", m.Value, "Okay");
        });
    }

    [RelayCommand]
    public async Task LoadTodoItemsAsync(){
        IsRefreshing = true;
        try {
            var items = await _todoService.GetAllTodoItemsAsync();
            TodoItems.Clear();
            foreach(var item in items) {
                TodoItems.Add(item);
            }
        } catch {
            //TODO Implement error handling
        } finally {
            IsRefreshing = false;
        }
    }

    [RelayCommand]
    async Task AddItem()
    {
        await Shell.Current.GoToAsync("/CreateItemPage");
    }

    [RelayCommand]
    async Task ItemTapped(TodoItem item)
    {
        if (item == null)
            return;

        await Shell.Current.GoToAsync($"/DetailsPage?id={item.Id}");
    }

    [RelayCommand]
    async Task MarkAsCompleted(TodoItem item)
    {
        if (item == null)
            return;

        item.Completed = true;
        await _todoService.UpdateTodoItemAsync(item);
        await LoadTodoItemsAsync();
    }

    [RelayCommand]
    async Task EditItem(TodoItem item)
    {
        if (item == null)
            return;

        await Shell.Current.GoToAsync($"/EditPage?id={item.Id}");
    }
}