using CommunityToolkit.Mvvm.Input;
using MicrosoftExampleProject.Models;

namespace MicrosoftExampleProject.PageModels;

public interface IProjectTaskPageModel
{
	IAsyncRelayCommand<ProjectTask> NavigateToTaskCommand { get; }
	bool IsBusy { get; }
}