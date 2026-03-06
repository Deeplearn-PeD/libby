#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHANGELOG="$PROJECT_ROOT/CHANGELOG.md"

cd "$PROJECT_ROOT"

get_current_version() {
    uv version 2>/dev/null | head -1 | awk '{print $2}'
}

get_previous_tag() {
    git describe --tags --abbrev=0 2>/dev/null || echo ""
}

generate_changelog() {
    local new_version=$1
    local prev_tag=$2
    local date_str=$(date +%Y-%m-%d)
    
    local commit_range
    if [[ -n "$prev_tag" ]]; then
        commit_range="$prev_tag..HEAD"
    else
        commit_range="HEAD"
    fi
    
    local commits feat_commits fix_commits docs_commits refactor_commits chore_commits other_commits
    
    feat_commits=$(git log $commit_range --pretty=format:"- %s" --grep="^feat" 2>/dev/null || echo "")
    fix_commits=$(git log $commit_range --pretty=format:"- %s" --grep="^fix" 2>/dev/null || echo "")
    docs_commits=$(git log $commit_range --pretty=format:"- %s" --grep="^docs" 2>/dev/null || echo "")
    refactor_commits=$(git log $commit_range --pretty=format:"- %s" --grep="^refactor" 2>/dev/null || echo "")
    chore_commits=$(git log $commit_range --pretty=format:"- %s" --grep="^chore" 2>/dev/null || echo "")
    
    other_commits=$(git log $commit_range --pretty=format:"- %s" --invert-grep --grep="^feat" --grep="^fix" --grep="^docs" --grep="^refactor" --grep="^chore" --all-match 2>/dev/null | grep -v "bump version" || echo "")
    
    local changelog_content=""
    changelog_content+="## [$new_version] - $date_str\n\n"
    
    if [[ -n "$feat_commits" ]]; then
        changelog_content+="### Features\n$feat_commits\n\n"
    fi
    
    if [[ -n "$fix_commits" ]]; then
        changelog_content+="### Bug Fixes\n$fix_commits\n\n"
    fi
    
    if [[ -n "$docs_commits" ]]; then
        changelog_content+="### Documentation\n$docs_commits\n\n"
    fi
    
    if [[ -n "$refactor_commits" ]]; then
        changelog_content+="### Refactoring\n$refactor_commits\n\n"
    fi
    
    if [[ -n "$chore_commits" ]]; then
        changelog_content+="### Chores\n$chore_commits\n\n"
    fi
    
    if [[ -n "$other_commits" ]]; then
        changelog_content+="### Other Changes\n$other_commits\n\n"
    fi
    
    echo -e "$changelog_content"
}

update_changelog_file() {
    local new_entry=$1
    
    if [[ ! -f "$CHANGELOG" ]]; then
        echo "# Changelog" > "$CHANGELOG"
        echo "" >> "$CHANGELOG"
        echo "All notable changes to this project will be documented in this file." >> "$CHANGELOG"
        echo "" >> "$CHANGELOG"
    fi
    
    local temp_file=$(mktemp)
    
    echo "# Changelog" > "$temp_file"
    echo "" >> "$temp_file"
    echo "All notable changes to this project will be documented in this file." >> "$temp_file"
    echo "" >> "$temp_file"
    echo -e "$new_entry" >> "$temp_file"
    
    if [[ -f "$CHANGELOG" ]]; then
        tail -n +5 "$CHANGELOG" >> "$temp_file" 2>/dev/null || true
    fi
    
    mv "$temp_file" "$CHANGELOG"
}

echo "=========================================="
echo "  Libby Release Script"
echo "=========================================="
echo

CURRENT_VERSION=$(get_current_version)
PREV_TAG=$(get_previous_tag)

echo "Current version: $CURRENT_VERSION"
if [[ -n "$PREV_TAG" ]]; then
    echo "Previous tag: $PREV_TAG"
fi
echo

echo "Git status:"
echo "----------------------------------------"
git status -s
echo "----------------------------------------"
echo

if [[ -z $(git status -s) ]]; then
    echo "No changes to commit."
    read -p "Do you want to bump version anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo "Select version bump type:"
echo "  1) patch - Bug fixes, small changes"
echo "  2) minor - New features, backward compatible"
echo "  3) major - Breaking changes"
echo
read -p "Enter choice [1-3]: " -n 1 -r choice
echo

case $choice in
    1) BUMP_TYPE="patch" ;;
    2) BUMP_TYPE="minor" ;;
    3) BUMP_TYPE="major" ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo
echo "Bumping version ($BUMP_TYPE)..."

uv version --bump "$BUMP_TYPE"

NEW_VERSION=$(get_current_version)
echo "New version: $NEW_VERSION"
echo

if [[ -n $(git status -s -- ':!.gitignore' 2>/dev/null || git status -s) ]]; then
    echo "Recent commits since $PREV_TAG:"
    echo "----------------------------------------"
    if [[ -n "$PREV_TAG" ]]; then
        git log "$PREV_TAG..HEAD" --oneline 2>/dev/null || git log --oneline -10
    else
        git log --oneline -10
    fi
    echo "----------------------------------------"
    echo
    
    read -p "Enter commit message (or press Enter for default): " commit_msg
    if [[ -z "$commit_msg" ]]; then
        commit_msg="chore: bump version to $NEW_VERSION"
    fi
fi

read -p "Proceed with release? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    git checkout pyproject.toml 2>/dev/null || true
    exit 1
fi

echo
echo "Generating changelog..."
CHANGELOG_ENTRY=$(generate_changelog "$NEW_VERSION" "$PREV_TAG")
update_changelog_file "$CHANGELOG_ENTRY"
echo "Changelog updated."

if [[ -n $(git status -s) ]]; then
    echo "Staging changes..."
    git add -A
    
    echo "Creating commit..."
    git commit -m "$commit_msg"
fi

echo "Creating git tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

echo
echo "=========================================="
echo "  Release v$NEW_VERSION created!"
echo "=========================================="
echo
echo "Changelog preview:"
echo "----------------------------------------"
echo -e "$CHANGELOG_ENTRY"
echo "----------------------------------------"
echo
echo "To push changes and tag:"
echo "  git push && git push --tags"
echo
echo "To undo (before pushing):"
echo "  git tag -d v$NEW_VERSION"
echo "  git reset --hard HEAD~1"
