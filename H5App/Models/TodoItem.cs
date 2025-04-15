using System.Diagnostics;

public class TodoItem {
    int Id {get; set;}
    string? Description {get; set;}
    DateTime CreatedTime {get; set;} = DateTime.UtcNow;
    PriorityLevel Priority {get; set;}
    bool Completed {get; set;} = false;
}

public enum PriorityLevel {
    Low,
    Normal,
    High
}