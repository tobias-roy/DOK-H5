using CommunityToolkit.Mvvm.ComponentModel;

namespace H5App.PageModels;
public partial class BasePageModel : ObservableObject
{
    [ObservableProperty]
    public partial bool IsBusy {get; set;} = false;

    [ObservableProperty]
    public partial string Title {get; set;}

    [ObservableProperty]
    public partial bool IsRefreshing {get; set;}   
}