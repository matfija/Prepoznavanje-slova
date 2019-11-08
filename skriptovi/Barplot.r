a <- c(789, 766, 736, 805, 768, 775, 773,
       734, 755, 747, 739, 761, 792, 783,
       753, 803, 783, 758, 748, 796, 813,
       764, 752, 787, 786, 734)

names(a) <- c('A', 'B', 'C', 'D', 'E', 'F', 'G',
              'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z')

barplot(a, main = 'Raspodela slova',
           xlab = 'Slova',
           ylab = 'Frekvencije',
           cex.names = .7,
           col = c('cadetblue'))
