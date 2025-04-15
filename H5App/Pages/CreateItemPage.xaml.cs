using H5App.PageModels;
namespace H5App.Pages;

public partial class CreateItemPage : ContentPage {
    public CreateItemPage(CreateTodoItemPageModel pageModel) {
        InitializeComponent();
        BindingContext = pageModel;
    }
}