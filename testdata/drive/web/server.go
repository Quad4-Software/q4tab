package web

import (
	_ "encoding/json"
	_ "net/http"
	_ "strings"
)

type Server struct {
	st *Store
}
