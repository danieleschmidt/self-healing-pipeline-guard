#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) generation script for Self-Healing Pipeline Guard.
Generates comprehensive dependency information for security and compliance.
"""

import json
import subprocess
import sys
import toml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import uuid


class SBOMGenerator:
    """Generates Software Bill of Materials (SBOM) in SPDX format."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.poetry_lock_path = project_root / "poetry.lock"
        
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate complete SBOM document."""
        sbom = {
            "SPDXID": "SPDXRef-DOCUMENT",
            "spdxVersion": "SPDX-2.3",
            "creationInfo": self._get_creation_info(),
            "name": "Self-Healing Pipeline Guard SBOM",
            "dataLicense": "CC0-1.0",
            "documentNamespace": f"https://github.com/danieleschmidt/self-healing-pipeline-guard/sbom/{uuid.uuid4()}",
            "packages": self._get_packages(),
            "relationships": self._get_relationships(),
            "externalDocumentRefs": []
        }
        
        return sbom
    
    def _get_creation_info(self) -> Dict[str, Any]:
        """Get creation information for the SBOM."""
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "creators": [
                "Tool: healing-guard-sbom-generator",
                "Organization: Terragon Labs"
            ],
            "licenseListVersion": "3.19"
        }
    
    def _get_packages(self) -> List[Dict[str, Any]]:
        """Get all packages including main application and dependencies."""
        packages = []
        
        # Add main application package
        packages.append(self._get_main_package())
        
        # Add Python dependencies
        packages.extend(self._get_python_dependencies())
        
        # Add system dependencies (if any)
        packages.extend(self._get_system_dependencies())
        
        return packages
    
    def _get_main_package(self) -> Dict[str, Any]:
        """Get main application package information."""
        pyproject_data = self._load_pyproject_toml()
        poetry_info = pyproject_data.get("tool", {}).get("poetry", {})
        
        return {
            "SPDXID": "SPDXRef-Package-HealingGuard",
            "name": poetry_info.get("name", "self-healing-pipeline-guard"),
            "versionInfo": poetry_info.get("version", "unknown"),
            "downloadLocation": poetry_info.get("homepage", "NOASSERTION"),
            "filesAnalyzed": True,
            "packageVerificationCode": {
                "packageVerificationCodeValue": self._calculate_package_hash()
            },
            "copyrightText": "Copyright (c) 2024 Terragon Labs",
            "licenseConcluded": poetry_info.get("license", "MIT"),
            "licenseDeclared": poetry_info.get("license", "MIT"),
            "supplier": "Organization: Terragon Labs",
            "originator": "Organization: Terragon Labs",
            "description": poetry_info.get("description", "AI-powered CI/CD failure detection and remediation"),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{poetry_info.get('name', 'self-healing-pipeline-guard')}@{poetry_info.get('version', 'unknown')}"
                }
            ]
        }
    
    def _get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Get Python dependencies from poetry.lock."""
        dependencies = []
        
        if not self.poetry_lock_path.exists():
            print("⚠️  poetry.lock not found, generating from pyproject.toml")
            return self._get_dependencies_from_pyproject()
        
        try:
            lock_data = toml.load(self.poetry_lock_path)
            packages = lock_data.get("package", [])
            
            for i, package in enumerate(packages):
                dep_info = {
                    "SPDXID": f"SPDXRef-Package-{package['name'].replace('-', '')}-{i}",
                    "name": package["name"],
                    "versionInfo": package["version"],
                    "downloadLocation": self._get_package_download_location(package),
                    "filesAnalyzed": False,
                    "copyrightText": "NOASSERTION",
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": "NOASSERTION",
                    "supplier": "NOASSERTION",
                    "description": package.get("description", ""),
                    "externalRefs": [
                        {
                            "referenceCategory": "PACKAGE-MANAGER",
                            "referenceType": "purl",
                            "referenceLocator": f"pkg:pypi/{package['name']}@{package['version']}"
                        }
                    ]
                }
                
                # Add checksums if available
                if "files" in package:
                    checksums = []
                    for file_info in package["files"]:
                        if "hash" in file_info:
                            algorithm, hash_value = file_info["hash"].split(":")
                            checksums.append({
                                "algorithm": algorithm.upper(),
                                "checksumValue": hash_value
                            })
                    
                    if checksums:
                        dep_info["checksums"] = checksums
                
                dependencies.append(dep_info)
        
        except Exception as e:
            print(f"❌ Error reading poetry.lock: {e}")
            return self._get_dependencies_from_pyproject()
        
        return dependencies
    
    def _get_dependencies_from_pyproject(self) -> List[Dict[str, Any]]:
        """Fallback: Get dependencies from pyproject.toml."""
        dependencies = []
        pyproject_data = self._load_pyproject_toml()
        
        poetry_deps = pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        
        for i, (name, version_spec) in enumerate(poetry_deps.items()):
            if name == "python":
                continue
                
            # Handle different version specification formats
            if isinstance(version_spec, dict):
                version = version_spec.get("version", "unknown")
            else:
                version = str(version_spec)
            
            dep_info = {
                "SPDXID": f"SPDXRef-Package-{name.replace('-', '')}-{i}",
                "name": name,
                "versionInfo": version,
                "downloadLocation": f"https://pypi.org/project/{name}/",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "supplier": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{name}@{version}"
                    }
                ]
            }
            
            dependencies.append(dep_info)
        
        return dependencies
    
    def _get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Get system-level dependencies (from Dockerfile)."""
        dependencies = []
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            return dependencies
        
        try:
            dockerfile_content = dockerfile_path.read_text()
            
            # Extract apt packages from RUN apt-get install commands
            import re
            apt_pattern = r'apt-get install[^\\]*?(?:\\[\s\S]*?)?(?=&&|\n|$)'
            apt_matches = re.findall(apt_pattern, dockerfile_content, re.MULTILINE)
            
            for i, match in enumerate(apt_matches):
                # Extract package names
                packages = re.findall(r'\b[a-z][a-z0-9\-\+\.]+', match)
                packages = [p for p in packages if p not in ['apt', 'get', 'install', 'update', 'upgrade']]
                
                for j, package in enumerate(packages):
                    dep_info = {
                        "SPDXID": f"SPDXRef-Package-system-{package}-{i}-{j}",
                        "name": package,
                        "versionInfo": "NOASSERTION",
                        "downloadLocation": "NOASSERTION",
                        "filesAnalyzed": False,
                        "copyrightText": "NOASSERTION",
                        "licenseConcluded": "NOASSERTION",
                        "licenseDeclared": "NOASSERTION",
                        "supplier": "NOASSERTION",
                        "packageType": "deb"
                    }
                    dependencies.append(dep_info)
        
        except Exception as e:
            print(f"⚠️  Error parsing Dockerfile: {e}")
        
        return dependencies
    
    def _get_relationships(self) -> List[Dict[str, Any]]:
        """Get package relationships."""
        relationships = [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-HealingGuard"
            }
        ]
        
        # Add dependency relationships
        pyproject_data = self._load_pyproject_toml()
        poetry_deps = pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        
        for i, name in enumerate(poetry_deps.keys()):
            if name == "python":
                continue
                
            relationships.append({
                "spdxElementId": "SPDXRef-Package-HealingGuard",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": f"SPDXRef-Package-{name.replace('-', '')}-{i}"
            })
        
        return relationships
    
    def _get_package_download_location(self, package: Dict[str, Any]) -> str:
        """Get download location for a package."""
        if "source" in package:
            source = package["source"]
            if source.get("type") == "git":
                return source.get("url", "NOASSERTION")
        
        return f"https://pypi.org/project/{package['name']}/"
    
    def _calculate_package_hash(self) -> str:
        """Calculate hash for the main package."""
        # Simple hash of pyproject.toml content
        if self.pyproject_path.exists():
            content = self.pyproject_path.read_bytes()
            return hashlib.sha1(content).hexdigest()
        
        return "unknown"
    
    def _load_pyproject_toml(self) -> Dict[str, Any]:
        """Load pyproject.toml file."""
        if not self.pyproject_path.exists():
            return {}
        
        try:
            return toml.load(self.pyproject_path)
        except Exception as e:
            print(f"❌ Error loading pyproject.toml: {e}")
            return {}
    
    def generate_vulnerability_report(self) -> Dict[str, Any]:
        """Generate vulnerability report using safety."""
        try:
            result = subprocess.run(
                ["poetry", "run", "safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"vulnerabilities": [], "status": "clean"}
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return {"vulnerabilities": vulnerabilities, "status": "issues_found"}
                except json.JSONDecodeError:
                    return {"vulnerabilities": [], "status": "error", "error": result.stderr}
        
        except Exception as e:
            return {"vulnerabilities": [], "status": "error", "error": str(e)}
    
    def save_sbom(self, output_path: Path, format: str = "json") -> None:
        """Save SBOM to file."""
        sbom = self.generate_sbom()
        
        if format.lower() == "json":
            output_path.write_text(json.dumps(sbom, indent=2, ensure_ascii=False))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📄 SBOM saved to {output_path}")
    
    def save_vulnerability_report(self, output_path: Path) -> None:
        """Save vulnerability report to file."""
        report = self.generate_vulnerability_report()
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"🔍 Vulnerability report saved to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBOM for Self-Healing Pipeline Guard")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("sbom.json"),
        help="Output file path (default: sbom.json)"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--vulnerability-report",
        "-v",
        type=Path,
        help="Generate vulnerability report to this file"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    generator = SBOMGenerator(args.project_root)
    
    print("🔧 Generating SBOM...")
    generator.save_sbom(args.output, args.format)
    
    if args.vulnerability_report:
        print("🔍 Generating vulnerability report...")
        generator.save_vulnerability_report(args.vulnerability_report)
    
    print("✅ SBOM generation completed!")


if __name__ == "__main__":
    main()