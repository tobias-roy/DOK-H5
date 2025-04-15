using H5App.PageModels;
namespace H5App.Pages;

public partial class EditPage : ContentPage {
    public EditPage(EditTodoItemPageModel pageModel) {
        InitializeComponent();
        BindingContext = pageModel;
    }
}