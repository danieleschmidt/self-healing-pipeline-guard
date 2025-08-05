{{/*
Expand the name of the chart.
*/}}
{{- define "healing-guard.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "healing-guard.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "healing-guard.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "healing-guard.labels" -}}
helm.sh/chart: {{ include "healing-guard.chart" . }}
{{ include "healing-guard.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: healing-guard
{{- end }}

{{/*
Selector labels
*/}}
{{- define "healing-guard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "healing-guard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "healing-guard.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "healing-guard.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the PostgreSQL password
*/}}
{{- define "healing-guard.postgresql.password" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.postgresql.auth.password }}
{{- else }}
{{- .Values.secrets.databasePassword }}
{{- end }}
{{- end }}

{{/*
Get the Redis password
*/}}
{{- define "healing-guard.redis.password" -}}
{{- if .Values.redis.enabled }}
{{- .Values.redis.auth.password }}
{{- else }}
{{- .Values.secrets.redisPassword }}
{{- end }}
{{- end }}

{{/*
PostgreSQL host
*/}}
{{- define "healing-guard.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "healing-guard.fullname" .) }}
{{- else }}
{{- "postgresql" }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "healing-guard.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "healing-guard.fullname" .) }}
{{- else }}
{{- "redis" }}
{{- end }}
{{- end }}