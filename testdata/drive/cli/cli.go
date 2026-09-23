package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func main() {
	add := flag.NewFlagSet("add", flag.ExitOnError)
	addText := add.String("text", "", "todo text")

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: todo <add|list|done>")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "add":
		add.Parse(os.Args[2:])
		if *addText == "" {
			fmt.Fprintln(os.Stderr, "add: -text required")
			os.Exit(2)
		}
		save(*addText)
	case "list":
		list()
	}
}

func save(text string) {
	f, err := os.OpenFile("todo.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "save:", err)
		os.Exit(1)
	}
	defer f.Close()
	json.NewEncoder(f).Encode(map[string]string{"text": text})
}

func list() {
	f, err := os.Open("todo.jsonl")
	if err != nil {
		return
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	for {
		var e map[string]string
		if err := dec.Decode(&e); err != nil {
			break
		}
		fmt.Println(e["text"])
	}
}
