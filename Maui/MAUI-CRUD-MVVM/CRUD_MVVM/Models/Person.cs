using CommunityToolkit.Mvvm.ComponentModel;

namespace CRUD_MVVM.Models;
public partial class Person : BaseModel
{
    [ObservableProperty]
    public partial string Name {get; set;}

    [ObservableProperty]
    public partial int Age{get; set;}
}
