using System.Globalization;
using H5App.Models;

namespace H5App.Converters;

public class PriorityToColorConverter : IValueConverter {
    public object Convert(object value, Type targetType, object parameter, CultureInfo cultureInfo) {
        if (value is PriorityLevel priority) {
            return priority switch {
                PriorityLevel.Low => Colors.Green,
                PriorityLevel.Normal => Colors.Orange,
                PriorityLevel.High => Colors.Red,
                _ => Colors.Gray,
            };
        }
        return Colors.Gray;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo cultureInfo){
        //TODO implement convert back functionality
        throw new NotImplementedException();
    }
}