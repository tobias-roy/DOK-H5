using H5App.PageModels;
namespace H5App.Pages;

public partial class DetailsPage : ContentPage{
    public DetailsPage(TodoItemDetailsPageModel pageModel) {
        InitializeComponent();
        BindingContext = pageModel;
    }
}