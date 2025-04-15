using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CRUD_MVVM.Models;
using CRUD_MVVM.Services;
using CRUD_MVVM.Views;

namespace CRUD_MVVM.ViewModels;

[QueryProperty(nameof(Person), "MyPerson")]
public partial class DetailsPageViewModel : BaseViewModel
{
    [ObservableProperty]
    public partial Person Person{get; set;}

    private readonly IDataService service;
    public DetailsPageViewModel(IDataService service)
    {
        this.service = service;
    }

    [RelayCommand]
    private async Task GoToAddEdit() {
        await Shell.Current.GoToAsync(nameof(AddEditPage), true, new Dictionary<string, object>
        {
            {"MyPerson", Person }
        });
    }

    [RelayCommand]
    private async Task Delete(){
        bool answer = await Shell.Current.DisplayAlert("DELETE?", "Are you sure?", "Ok", "Cancel");
        if (answer)
        {
            service.DeletePerson(Person);
        }
        await Shell.Current.GoToAsync("..");
    }
}
