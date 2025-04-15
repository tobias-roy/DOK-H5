using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using H5App.Models;
using H5App.Services;

namespace H5App.PageModels;

public partial class CreateTodoItemPageModel : BasePageModel
    {
        private readonly ITodoService _todoService;

        [ObservableProperty]
        private string _description;

        [ObservableProperty]
        private PriorityLevel _priority = PriorityLevel.Normal;

        public Array PriorityLevels => Enum.GetValues(typeof(PriorityLevel));

        public CreateTodoItemPageModel(ITodoService todoService)
        {
            _todoService = todoService;
        }

        [RelayCommand]
        async Task Save()
        {
            if (string.IsNullOrWhiteSpace(Description))
                return;

            var newItem = new TodoItem
            {
                Description = Description,
                Priority = Priority,
                CreatedTime = DateTime.Now,
                Completed = false
            };

            await _todoService.AddTodoItemAsync(newItem);
            await Cancel();
        }

        [RelayCommand]
        async Task Cancel()
        {
            // Reset the fields
            Description = string.Empty;
            Priority = PriorityLevel.Normal;

            // Navigate back
            await Shell.Current.GoToAsync("..");
        }
    }