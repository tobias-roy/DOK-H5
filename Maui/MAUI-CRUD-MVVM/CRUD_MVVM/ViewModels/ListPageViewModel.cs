using CommunityToolkit.Mvvm.Input;
using CRUD_MVVM.Models;
using CRUD_MVVM.Services;
using CRUD_MVVM.Views;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows.Input;

namespace CRUD_MVVM.ViewModels;
public partial class ListPageViewModel : BaseViewModel
{
    private readonly IDataService service;
    public ListPageViewModel(IDataService service)
    {
        this.service = service;
    }
    public ObservableCollection<Person> Persons { get; } = new();

    [RelayCommand]
    private async Task GetPerson(){
        await GetPersonsAsync();
    }

    private async Task GetPersonsAsync()
    {
        if (IsBusy)
            return;
        try
        {
            IsBusy = true;

            List<Person> persons = service.GetPersons();

            if (Persons.Count != 0)
                Persons.Clear();

            foreach (Person person in persons)
                Persons.Add(person);

        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Unable to get persons: {ex.Message}");
            await Shell.Current.DisplayAlert("Error!", ex.Message, "OK");
        }
        finally
        {
            IsBusy = false;
            IsRefreshing = false;
        }
    }

    [RelayCommand]
    private async Task GoToDetails(Person person){
        if (person == null)
            return;

        await Shell.Current.GoToAsync(nameof(DetailsPage), true, new Dictionary<string, object>
        {
            {"MyPerson", person }
        });
    }

    [RelayCommand]
    private async Task GoToAddEdit(){
        await Shell.Current.GoToAsync(nameof(AddEditPage), true, new Dictionary<string, object>
        {
            {"MyPerson", new Person() }
        });
    }
}