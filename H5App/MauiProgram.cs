using H5App.PageModels;
using H5App.Pages;
using H5App.Services;
using Microsoft.Extensions.Logging;

namespace H5App;

public static class MauiProgram
{
	public static MauiApp CreateMauiApp()
	{
		var builder = MauiApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

		builder.Services.AddSingleton<ITodoService, TodoService>();
		builder.Services.AddTransient<TodoListPageModel>();
		builder.Services.AddTransient<CreateTodoItemPageModel>();
		builder.Services.AddTransient<TodoItemDetailsPageModel>();
		builder.Services.AddTransient<EditTodoItemPageModel>();

		// Register Pages
		builder.Services.AddTransient<MainPage>();
		builder.Services.AddTransient<CreateItemPage>();
		builder.Services.AddTransient<DetailsPage>();
		builder.Services.AddTransient<EditPage>();

#if DEBUG
		builder.Logging.AddDebug();
#endif

		return builder.Build();
	}
}
