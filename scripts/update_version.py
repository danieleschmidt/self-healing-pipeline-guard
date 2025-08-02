#!/usr/bin/env python3
"""
Version update script for Self-Healing Pipeline Guard.
Updates version across all relevant files during release process.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


class VersionUpdater:
    """Updates version across multiple files in the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.files_to_update = {
            "pyproject.toml": self.update_pyproject_toml,
            "package.json": self.update_package_json,
            "healing_guard/__init__.py": self.update_python_init,
            "Dockerfile": self.update_dockerfile,
            "docker-compose.yml": self.update_docker_compose,
            "docs/conf.py": self.update_docs_config,
            "helm/Chart.yaml": self.update_helm_chart,
            "mkdocs.yml": self.update_mkdocs_config
        }
    
    def update_version(self, new_version: str) -> Dict[str, bool]:
        """Update version in all relevant files."""
        results = {}
        
        print(f"🔄 Updating version to {new_version}")
        
        for file_path, update_func in self.files_to_update.items():
            full_path = self.project_root / file_path
            
            if full_path.exists():
                try:
                    success = update_func(full_path, new_version)
                    results[file_path] = success
                    status = "✅" if success else "❌"
                    print(f"  {status} {file_path}")
                except Exception as e:
                    results[file_path] = False
                    print(f"  ❌ {file_path}: {e}")
            else:
                results[file_path] = None
                print(f"  ⏭️  {file_path} (not found)")
        
        return results
    
    def update_pyproject_toml(self, file_path: Path, version: str) -> bool:
        """Update version in pyproject.toml."""
        content = file_path.read_text()
        
        # Update version in [tool.poetry] section
        updated_content = re.sub(
            r'(version\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{version}"',
            content
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_package_json(self, file_path: Path, version: str) -> bool:
        """Update version in package.json."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if data.get('version') != version:
                data['version'] = version
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write('\n')  # Add trailing newline
                
                return True
            
            return False
        
        except (json.JSONDecodeError, KeyError):
            return False
    
    def update_python_init(self, file_path: Path, version: str) -> bool:
        """Update version in Python __init__.py file."""
        content = file_path.read_text()
        
        # Look for __version__ assignment
        updated_content = re.sub(
            r'(__version__\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{version}"',
            content
        )
        
        # If no __version__ found, add it
        if '__version__' not in content:
            updated_content = f'__version__ = "{version}"\n' + content
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_dockerfile(self, file_path: Path, version: str) -> bool:
        """Update version in Dockerfile LABEL."""
        content = file_path.read_text()
        
        updated_content = re.sub(
            r'(LABEL\s+version=)["\'][^"\']+["\']',
            f'\\1"{version}"',
            content
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_docker_compose(self, file_path: Path, version: str) -> bool:
        """Update version in docker-compose.yml image tags."""
        content = file_path.read_text()
        
        # Update image tags that include version
        updated_content = re.sub(
            r'(image:\s*[^:]+:)v?[0-9]+\.[0-9]+\.[0-9]+[^"\s]*',
            f'\\1v{version}',
            content
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_docs_config(self, file_path: Path, version: str) -> bool:
        """Update version in Sphinx docs configuration."""
        content = file_path.read_text()
        
        # Update version and release variables
        updated_content = re.sub(
            r'(version\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{version}"',
            content
        )
        
        updated_content = re.sub(
            r'(release\s*=\s*)["\'][^"\']+["\']',
            f'\\1"{version}"',
            updated_content
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_helm_chart(self, file_path: Path, version: str) -> bool:
        """Update version in Helm Chart.yaml."""
        content = file_path.read_text()
        
        # Update both version and appVersion
        updated_content = re.sub(
            r'(^version:\s*)[0-9]+\.[0-9]+\.[0-9]+.*$',
            f'\\1{version}',
            content,
            flags=re.MULTILINE
        )
        
        updated_content = re.sub(
            r'(^appVersion:\s*)["\']?[0-9]+\.[0-9]+\.[0-9]+.*["\']?$',
            f'\\1"{version}"',
            updated_content,
            flags=re.MULTILINE
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def update_mkdocs_config(self, file_path: Path, version: str) -> bool:
        """Update version in MkDocs configuration."""
        content = file_path.read_text()
        
        # Update site_description or extra.version if present
        updated_content = re.sub(
            r'(site_description:.*version\s+)[0-9]+\.[0-9]+\.[0-9]+',
            f'\\1{version}',
            content
        )
        
        # Update extra.version if present
        updated_content = re.sub(
            r'(version:\s*)["\']?[0-9]+\.[0-9]+\.[0-9]+["\']?',
            f'\\1"{version}"',
            updated_content
        )
        
        if content != updated_content:
            file_path.write_text(updated_content)
            return True
        
        return False
    
    def validate_version_format(self, version: str) -> bool:
        """Validate version follows semantic versioning."""
        # Semantic versioning pattern
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        return bool(re.match(pattern, version))
    
    def create_version_summary(self, version: str, results: Dict[str, bool]) -> str:
        """Create a summary of version update results."""
        summary_lines = [
            f"Version Update Summary: {version}",
            "=" * 50
        ]
        
        successful_updates = []
        failed_updates = []
        skipped_files = []
        
        for file_path, result in results.items():
            if result is True:
                successful_updates.append(file_path)
            elif result is False:
                failed_updates.append(file_path)
            else:  # result is None
                skipped_files.append(file_path)
        
        if successful_updates:
            summary_lines.append(f"\n✅ Successfully updated ({len(successful_updates)}):")
            for file_path in successful_updates:
                summary_lines.append(f"  - {file_path}")
        
        if failed_updates:
            summary_lines.append(f"\n❌ Failed to update ({len(failed_updates)}):")
            for file_path in failed_updates:
                summary_lines.append(f"  - {file_path}")
        
        if skipped_files:
            summary_lines.append(f"\n⏭️  Skipped (not found) ({len(skipped_files)}):")
            for file_path in skipped_files:
                summary_lines.append(f"  - {file_path}")
        
        return "\n".join(summary_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update version across project files")
    parser.add_argument("version", help="New version to set (e.g., 1.2.3)")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print detailed summary of updates"
    )
    
    args = parser.parse_args()
    
    updater = VersionUpdater(args.project_root)
    
    # Validate version format
    if not updater.validate_version_format(args.version):
        print(f"❌ Invalid version format: {args.version}")
        print("   Expected format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]")
        sys.exit(1)
    
    if args.dry_run:
        print(f"🔍 Dry run: Would update version to {args.version}")
        for file_path in updater.files_to_update.keys():
            full_path = args.project_root / file_path
            status = "📁" if full_path.exists() else "❓"
            print(f"  {status} {file_path}")
        sys.exit(0)
    
    # Perform version update
    results = updater.update_version(args.version)
    
    # Check if any updates failed
    failed_count = sum(1 for result in results.values() if result is False)
    success_count = sum(1 for result in results.values() if result is True)
    
    if args.summary:
        print(f"\n{updater.create_version_summary(args.version, results)}")
    
    print(f"\n📊 Summary: {success_count} updated, {failed_count} failed")
    
    if failed_count > 0:
        print("⚠️  Some files failed to update. Please check manually.")
        sys.exit(1)
    else:
        print("🎉 All applicable files updated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()