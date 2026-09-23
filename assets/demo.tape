# Regenerate: create assets/demo/ with an empty .git marker dir,
# main.go below, and init.lua that prepends nvim/ to runtimepath and
# calls require("q4tab").setup(). Then: vhs assets/demo.tape
#
Output assets/demo.gif

Set Shell "bash"
Set FontSize 22
Set Width 1400
Set Height 750
Set Padding 20
Set TypingSpeed 45ms
Set Framerate 30
Set Theme "Catppuccin Mocha"

Hide
Type "cd /run/media/user1/projects/shipper/q4tab/assets/demo"
Enter
Type "nvim -u init.lua main.go"
Enter
Show
Sleep 6.5s

# inside func main, blank line
Type "5ji"
Sleep 300ms
Type "fmt.Pri"
Sleep 3.5s
Tab
Sleep 600ms

Escape
Type "o"
Type "for i :="
Sleep 3s
Tab
Sleep 1.5s

Hide
Escape
Type ":q!"
Enter
