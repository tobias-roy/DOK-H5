using CommunityToolkit.Mvvm.ComponentModel;
namespace H5App.PageModels;

public partial class BoilerplateModel : ObservableObject
{
	private bool _isNavigatedTo;
	private bool _dataLoaded;

	[ObservableProperty]
	bool _isBusy;

	[ObservableProperty]
	bool _isRefreshing;

	[ObservableProperty]
	private string _today = DateTime.Now.ToString("dddd, MMM d");


	public BoilerplateModel()
	{
	}

	// private async Task LoadData()
	// {
	// 	try
	// 	{
	// 		IsBusy = true;
    //         //Load some data
	// 	}
	// 	finally
	// 	{
	// 		IsBusy = false;
	// 		OnPropertyChanged(nameof());
	// 	}
	// }


	// [RelayCommand]
	// private Task AddTask()
	// 	=> Shell.Current.GoToAsync($"task");

	// [RelayCommand]
	// private Task NavigateToTask(ProjectTask task)
	// 	=> Shell.Current.GoToAsync($"task?id={task.ID}");
}