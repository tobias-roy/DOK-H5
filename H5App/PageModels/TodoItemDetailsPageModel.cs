using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using H5App.Models;
using H5App.Services;

namespace H5App.PageModels;

[QueryProperty(nameof(ItemId), "id")]
    public partial class TodoItemDetailsPageModel : BasePageModel
    {
        private readonly ITodoService _todoService;

        [ObservableProperty]
        private TodoItem _todoItem;

        [ObservableProperty]
        private int _itemId;

        public TodoItemDetailsPageModel(ITodoService todoService)
        {
            _todoService = todoService;
        }

        [RelayCommand]
        async Task LoadItem()
        {
            TodoItem = await _todoService.GetTodoItemByIdAsync(ItemId);
        }

        [RelayCommand]
        async Task GoToEdit()
        {
            await Shell.Current.GoToAsync($"/EditPage?id={ItemId}");
        }

        [RelayCommand]
        async Task GoBack()
        {
            await Shell.Current.GoToAsync("..");
        }

        partial void OnItemIdChanged(int value)
        {
            LoadItemCommand.Execute(null);
        }
    }