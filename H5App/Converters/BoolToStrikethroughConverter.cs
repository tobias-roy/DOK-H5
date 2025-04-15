using System.Globalization;
using H5App.Models;

namespace H5App.Converters;

public class BoolToStrikeThroughConverter : IValueConverter {
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isCompleted && isCompleted)
            {
                return TextDecorations.Strikethrough;
            }
            return TextDecorations.None;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
}