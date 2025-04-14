# App programming 3 - H5

This is a repository of the assignments during my H5 course - specifically for the class App programming 3.

## Shortcuts

[Worklong](#worklog)

[Notes](#notes)

# Worklog

## Day 1 - Repetition


# Notes
### General

The new shit for a viewmodel is to : OjservableObject and then create shit as [ObservableProperty] public partial string "sometime" {get; set;}

When creatiung commands you should create them as [RelayCmmand] private void ThisCommand() {do something}

private global;;CommunityToolkit.Mvvm,Inoput.RelayCommand? syncUICommand; is the backing field.

A relay command with a parameter CanExecute nameof(Method) checks first to see if the method returns true or false, if false the method wont execute.

Too survey and make it reactive you need to have notifycanexecutechangedfor on the observable property.

NotifyPropertyChangedFor(nameof(variable))



# Project
