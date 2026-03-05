#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEYS_DIR="$SCRIPT_DIR/../ssh_keys"

mkdir -p "$SSH_KEYS_DIR"
cd "$SSH_KEYS_DIR"

echo "Setting up SSH keys for Libby SFTP server..."
echo "Keys directory: $SSH_KEYS_DIR"

if [ -f "authorized_keys" ]; then
    echo "authorized_keys already exists. Skipping key generation."
    echo "If you want to regenerate, delete the authorized_keys file first."
    exit 0
fi

echo "Generating ED25519 SSH key pair..."
ssh-keygen -t ed25519 -f ssh_host_ed25519_key -N "" -C "libby-sftp"

echo "Creating authorized_keys from public key..."
cp ssh_host_ed25519_key.pub authorized_keys

echo "Setting permissions..."
chmod 600 ssh_host_ed25519_key
chmod 644 ssh_host_ed25519_key.pub
chmod 600 authorized_keys

echo ""
echo "SSH keys setup complete!"
echo ""
echo "Private key (keep secure): $SSH_KEYS_DIR/ssh_host_ed25519_key"
echo "Public key: $SSH_KEYS_DIR/ssh_host_ed25519_key.pub"
echo "Authorized keys: $SSH_KEYS_DIR/authorized_keys"
echo ""
echo "To connect to the SFTP server:"
echo "  sftp -i $SSH_KEYS_DIR/ssh_host_ed25519_key -P 2222 libby@localhost"
echo ""
echo "To upload a file:"
echo "  sftp -i $SSH_KEYS_DIR/ssh_host_ed25519_key -P 2222 libby@localhost"
echo "  sftp> put document.pdf"
echo "  sftp> bye"
