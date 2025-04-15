using System.ComponentModel;
using System.Runtime.CompilerServices;
using CommunityToolkit.Mvvm.ComponentModel;

namespace CRUD_MVVM.ViewModels;
public partial class BaseViewModel : ObservableObject
{
    [ObservableProperty]
    public partial bool IsBusy {get; set;} = false;

    [ObservableProperty]
    public partial string Title {get; set;}

    [ObservableProperty]
    public partial bool IsRefreshing {get; set;}   
}