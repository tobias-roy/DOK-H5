using System.ComponentModel;
using System.Runtime.CompilerServices;
using CommunityToolkit.Mvvm.ComponentModel;

namespace CRUD_MVVM.Models;
public partial class BaseModel : ObservableObject
{
    [ObservableProperty]
    public partial int Id { get; set; }
}
