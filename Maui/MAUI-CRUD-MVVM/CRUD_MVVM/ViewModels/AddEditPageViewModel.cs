using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CRUD_MVVM.Models;
using CRUD_MVVM.Services;

namespace CRUD_MVVM.ViewModels;

[QueryProperty(nameof(Person), "MyPerson")]
public partial class AddEditPageViewModel : BaseViewModel
{
    private readonly IDataService service;
    public AddEditPageViewModel(IDataService service)
    {
        this.service = service;
    }

    [ObservableProperty]
    public partial string Mode { get; set; }

    [ObservableProperty]
    public partial Person Person{get; set;}

    [RelayCommand]
    private void MakeOlder(){
        Person.Age++;
    }

    [RelayCommand]
    private async Task Save(){
        service.SavePerson(Person);
        await Shell.Current.GoToAsync("..");
    }
}
