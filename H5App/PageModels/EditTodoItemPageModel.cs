using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using H5App.Messages;
using H5App.Models;
using H5App.Services;

namespace H5App.PageModels;

[QueryProperty(nameof(ItemId), "id")]
    public partial class EditTodoItemPageModel : BasePageModel
    {
        private readonly ITodoService _todoService;

        [ObservableProperty]
        private TodoItem _todoItem;

        [ObservableProperty]
        private int _itemId;

        [ObservableProperty]
        private string _description; 

        [ObservableProperty]
        private PriorityLevel _priority;

        [ObservableProperty]
        private bool _completed;

        public Array PriorityLevels => Enum.GetValues(typeof(PriorityLevel));

        public EditTodoItemPageModel(ITodoService todoService)
        {
            _todoService = todoService;
        }

        [RelayCommand]
        async Task LoadItem()
        {
            TodoItem = await _todoService.GetTodoItemByIdAsync(ItemId);
            if (TodoItem != null)
            {
                Description = TodoItem.Description;
                Priority = TodoItem.Priority;
                Completed = TodoItem.Completed;
            }
        }

        [RelayCommand]
        async Task Save()
        {
            if (string.IsNullOrWhiteSpace(Description) || TodoItem == null)
                return;

            TodoItem.Description = Description;
            TodoItem.Priority = Priority;
            TodoItem.Completed = Completed;

            await _todoService.UpdateTodoItemAsync(TodoItem);
            await Shell.Current.GoToAsync("..");
        }

        [RelayCommand]
        async Task Delete()
        {
            if (TodoItem == null)
                return;

            bool confirmed = await Shell.Current.DisplayAlert(
                "Delete Item",
                $"Are you sure you want to delete '{TodoItem.Description}'?",
                "Yes", "No");
            if (confirmed)
            {
                await _todoService.DeleteTodoItemAsync(TodoItem.Id);
                //Send message with the messenger:
                await Shell.Current.GoToAsync("//MainPage");
                WeakReferenceMessenger.Default.Send(new DisplayMessage(TodoItem.Description));
            }
        }

        [RelayCommand]
        async Task Cancel()
        {
            await Shell.Current.GoToAsync("..");
        }

        partial void OnItemIdChanged(int value)
        {
            LoadItemCommand.Execute(null);
        }
    }