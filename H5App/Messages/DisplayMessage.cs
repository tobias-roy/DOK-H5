using CommunityToolkit.Mvvm.Messaging.Messages;

namespace H5App.Messages;

public class DisplayMessage : ValueChangedMessage<string>
{
    public DisplayMessage(string value) : base(value)
    {
        
    }
}