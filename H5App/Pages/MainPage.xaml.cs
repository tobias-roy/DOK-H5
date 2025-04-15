using H5App.PageModels;

namespace H5App.Pages
{
    public partial class MainPage : ContentPage
    {
        private TodoListPageModel _pageModel;

        public MainPage(TodoListPageModel pageModel)
        {
            InitializeComponent();
            _pageModel = pageModel;
            BindingContext = _pageModel;
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            _pageModel.LoadTodoItemsCommand.Execute(null);
        }
    }
}