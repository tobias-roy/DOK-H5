using MicrosoftExampleProject.Models;
using MicrosoftExampleProject.PageModels;

namespace MicrosoftExampleProject.Pages;

public partial class MainPage : ContentPage
{
	public MainPage(MainPageModel model)
	{
		InitializeComponent();
		BindingContext = model;
	}
}