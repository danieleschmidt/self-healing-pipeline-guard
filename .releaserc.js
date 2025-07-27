// Semantic Release configuration for Self-Healing Pipeline Guard
// Automates version management and release process

module.exports = {
  branches: [
    'main',
    {
      name: 'develop',
      prerelease: 'beta'
    },
    {
      name: 'release/*',
      prerelease: 'rc'
    }
  ],
  
  plugins: [
    // Analyze commits to determine the type of release
    [
      '@semantic-release/commit-analyzer',
      {
        preset: 'angular',
        releaseRules: [
          { type: 'feat', release: 'minor' },
          { type: 'fix', release: 'patch' },
          { type: 'perf', release: 'patch' },
          { type: 'revert', release: 'patch' },
          { type: 'docs', release: false },
          { type: 'style', release: false },
          { type: 'chore', release: false },
          { type: 'refactor', release: 'patch' },
          { type: 'test', release: false },
          { type: 'build', release: false },
          { type: 'ci', release: false },
          { breaking: true, release: 'major' }
        ],
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES']
        }
      }
    ],
    
    // Generate release notes
    [
      '@semantic-release/release-notes-generator',
      {
        preset: 'angular',
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        },
        writerOpts: {
          commitsSort: ['subject', 'scope']
        },
        presetConfig: {
          types: [
            { type: 'feat', section: '🚀 Features' },
            { type: 'fix', section: '🐛 Bug Fixes' },
            { type: 'perf', section: '⚡ Performance Improvements' },
            { type: 'revert', section: '🔄 Reverts' },
            { type: 'docs', section: '📚 Documentation', hidden: false },
            { type: 'style', section: '💄 Styles', hidden: true },
            { type: 'chore', section: '🔧 Chores', hidden: true },
            { type: 'refactor', section: '♻️ Code Refactoring' },
            { type: 'test', section: '✅ Tests', hidden: true },
            { type: 'build', section: '📦 Build System', hidden: true },
            { type: 'ci', section: '👷 CI/CD', hidden: true }
          ]
        }
      }
    ],
    
    // Update version in pyproject.toml
    [
      '@semantic-release/exec',
      {
        verifyReleaseCmd: 'echo "Verifying release ${nextRelease.version}"',
        prepareCmd: 'poetry version ${nextRelease.version}',
        publishCmd: 'echo "Publishing version ${nextRelease.version}"'
      }
    ],
    
    // Update CHANGELOG.md
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: '# Changelog\n\nAll notable changes to this project will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).'
      }
    ],
    
    // Commit updated files
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'pyproject.toml',
          'package.json',
          'package-lock.json'
        ],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ],
    
    // Create GitHub release
    [
      '@semantic-release/github',
      {
        assets: [
          {
            path: 'dist/*.whl',
            label: 'Python Wheel'
          },
          {
            path: 'dist/*.tar.gz',
            label: 'Source Distribution'
          },
          {
            path: 'artifacts/deployment-*.tar.gz',
            label: 'Deployment Package'
          },
          {
            path: 'artifacts/config-*.tar.gz',
            label: 'Configuration Package'
          },
          {
            path: 'reports/security-report.json',
            label: 'Security Report'
          }
        ],
        successComment: false,
        failComment: false,
        failTitle: false,
        labels: false,
        releasedLabels: false,
        addReleases: 'bottom'
      }
    ],
    
    // Notify Slack
    [
      '@semantic-release/slack-webhook',
      {
        notifyOnSuccess: true,
        notifyOnFail: true,
        onSuccessTemplate: {
          text: '🎉 A new version of Self-Healing Pipeline Guard has been released!\n\n*Version:* $npm_package_version\n*Repository:* $repo_url\n*Commit:* $repo_commit\n\n*Release Notes:*\n$release_notes'
        },
        onFailTemplate: {
          text: '❌ Release failed for Self-Healing Pipeline Guard\n\n*Repository:* $repo_url\n*Commit:* $repo_commit\n*Error:* $error_message'
        }
      }
    ]
  ],
  
  // Release configuration
  preset: 'angular',
  
  // Verify conditions before release
  verifyConditions: [
    '@semantic-release/changelog',
    '@semantic-release/git',
    '@semantic-release/github'
  ],
  
  // Additional configuration
  tagFormat: 'v${version}',
  
  // Environment variables required
  env: {
    GITHUB_TOKEN: process.env.GITHUB_TOKEN,
    SLACK_WEBHOOK: process.env.SLACK_WEBHOOK_URL,
    NPM_TOKEN: process.env.NPM_TOKEN
  }
};