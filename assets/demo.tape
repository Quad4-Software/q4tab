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
Set TypingSpeed 40ms
Set Framerate 30
Set Theme "Catppuccin Mocha"

Hide
Type "cd /run/media/user1/projects/shipper/q4tab/assets/demo"
Enter
Type "nvim -u init.lua main.go"
Enter
# LSP attach + model warmup, hidden so the gif starts ready
Sleep 6s
Type "12j"
Show
Sleep 400ms

# beat 1: defer
Type "i"
Type "defer f."
Sleep 1.1s
Tab
Sleep 500ms

# beat 2: fmt call
Escape
Type "o"
Type "fmt."
Sleep 1.1s
Tab
Sleep 500ms

# beat 3: loop with trailing multi-line ghost
Escape
Type "o"
Type "for i :="
Sleep 1.1s
Tab
Sleep 1.5s

Hide
Escape
Type ":q!"
Enter
