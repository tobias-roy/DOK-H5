namespace H5App.Models;

public class TodoItem {
    public int Id {get; set;}
    public string? Description {get; set;}
    public DateTime CreatedTime {get; set;} = DateTime.UtcNow;
    public PriorityLevel Priority {get; set;}
    public bool Completed {get; set;} = false;
}