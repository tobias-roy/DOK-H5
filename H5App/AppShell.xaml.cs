namespace H5App;

public partial class AppShell : Shell
{
	public AppShell()
	{
		InitializeComponent();

		Routing.RegisterRoute("CreateItemPage", typeof(Pages.CreateItemPage));
		Routing.RegisterRoute("DetailsPage", typeof(Pages.DetailsPage));
		Routing.RegisterRoute("EditPage", typeof(Pages.EditPage));
	}
}
